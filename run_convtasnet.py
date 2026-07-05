"""
End-to-end wrapper for the IC Conv-TasNet speech enhancement pipeline.

Orchestrates four sequential phases — generate, preprocess, train, test — for
both baseline (microphone input) and AmbiDrop (real-ACN Ambisonics) modes.
AmbiDrop test bypasses preprocessing and runs MicToRealAmbisonicsDataset on-the-fly.

Public interface:
    generate — synthesise Type A/B/C raw data from speech and room simulation
    preprocess — convert raw ex_N/ folders to .pt files (STFT for baseline, real-ACN for AmbiDrop)
    train — train Conv-TasNet model(s) via Solver and save checkpoints
    test — evaluate trained model(s) and print SI-SDR per array

Phases (all enabled by default, skip any with --actions):
  generate    → synthesise raw data from speech + room simulation
  preprocess  → convert raw data to .pt files (mic STFT for baseline,
                real-ACN time-domain for AmbiDrop training)
  train       → train Conv-TasNet model(s) via the Solver class
  test        → evaluate trained model(s) and print SI-SDR

Array configuration
  ARRAYS_TRAIN  — arrays merged into a single flat training/val dataset.
  ARRAYS_TEST   — arrays evaluated separately at test time.
                  Need not overlap with ARRAYS_TRAIN.

Note: AmbiDrop test bypasses preprocessing entirely — it uses
MicToRealAmbisonicsDataset which computes real-ACN Ambisonics on-the-fly
from raw p.wav + the array's steering matrix, exactly mirroring deployment.

Usage:
    # First time — generate everything, then train and evaluate:
    python run_convtasnet.py --mode both --actions generate preprocess train test

    # Raw data already exists, skip generation:
    python run_convtasnet.py --mode ambidrop --actions preprocess train test

    # Evaluate from an existing checkpoint (no training):
    python run_convtasnet.py --mode ambidrop --actions test \
        --checkpoint checkpoints/run_convtasnet/ambidrop/final.pth.tar
"""

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

import ConvTasNet.model as tasnet_model
from ConvTasNet.solver import Solver
from ConvTasNet.datasets import SimDS_preprocessed, MicToRealAmbisonicsDataset, PrecomputedASMDataset
from ambidrop.preprocess import (
    preprocess_mic, preprocess_sh_time,
    preprocess_dataset, preprocess_dataset_multi,
)
from ambidrop.losses import si_snr
from ambidrop.constants import get_device, REF_IDX_MAP
from ambidrop.signal_utils import find_ref_mic

try:
    from datagenerator.helpers import build_array, RigidSphereArrayConfig
    from datagenerator.generate_ambidrop_train_ds import generate_dataset as gen_ambidrop
    from datagenerator.generate_baseline_train_ds import (
        generate_dataset as gen_baseline,
        ArraySpec,
    )
    from datagenerator.generate_inference_ds import generate_dataset as gen_test
    _DATAGENERATOR_AVAILABLE = True
except ImportError:
    _DATAGENERATOR_AVAILABLE = False
    ArraySpec = None
    RigidSphereArrayConfig = None


# ============================================================
# === USER CONFIG — edit these before running ================
# ============================================================

# ── Speech data paths ─────────────────────────────────────────────────────────
# Point WSJ0_ROOT at the root of your WSJ0 corpus.
# Expected sub-directories: si_tr_s (train), si_dt_05 (val), si_et_05 (test).
WSJ0_ROOT    = "/Users/mikitatarjitzky/Documents/speech enhancement - ACL/wsj0"
SPEECH_TRAIN = os.path.join(WSJ0_ROOT, "si_tr_s")
SPEECH_VAL   = os.path.join(WSJ0_ROOT, "si_dt_05")
SPEECH_TEST  = os.path.join(WSJ0_ROOT, "si_et_05")

# ── Output directories ────────────────────────────────────────────────────────
DATA_ROOT = "datasets/run_convtasnet"  # raw + preprocessed data go here
CKPT_DIR  = "checkpoints/ConvTasNet"  # newly trained checkpoints saved here

# ── Pre-existing evaluation datasets ─────────────────────────────────────────
# Used by default when --actions test is run without generating new data.
# Set to None to force evaluation on freshly generated data instead.
EVAL_TRAIN_DIR = "datasets/experiment_full_anm/test_of_train_ds_preprocessed"
EVAL_TEST_DIR  = "datasets/experiment_full_anm/test_of_test_ds_preprocessed"
RAW_TRAIN_DIR  = "datasets/experiment_full_anm/test_of_train_ds"
RAW_TEST_DIR   = "datasets/experiment_full_anm/test_of_test_ds"

# ── Preferred checkpoints for --actions test (inside CKPT_DIR) ───────────────
# After training via this script the newly trained model takes priority.
CKPT_BASELINE = "run_2026-04-09_10-55/final.pth.tar"
CKPT_AMBIDROP = "run_2026-04-09_08-35/final.pth.tar"

# ── Dataset sizes ─────────────────────────────────────────────────────────────
N_TRAIN = 200
N_VAL   = 40
N_TEST  = 40

FS = 16000

# ── Model architecture ────────────────────────────────────────────────────────
ENC_DIM     = 512
FEATURE_DIM = 128
CH_DIM      = 8
WIN         = 16
LAYER       = 8
STACK       = 1
KERNEL      = 3
NUM_SPK     = 1
CAUSAL      = False

DROP_TYPE = "SHChannelDropout"
DROP_PROB = 0.4
MAX_DROP  = 3

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 8
LR         = 1e-3

# ── Array configurations ──────────────────────────────────────────────────────
# ARRAYS_TRAIN: merged into one flat training dataset.
# ARRAYS_TEST:  each array evaluated separately (can include unseen geometries).

N_MICS = 7
SOURCE_GRID_POINTS = 480
ARRAY_DURATION_SEC = 0.008  # used to build steering matrix for on-the-fly ASM

if _DATAGENERATOR_AVAILABLE:
    from shroom.geometry.sampling import sphereicalGrid  # noqa: lazy import
    RIGID_SPHERE_MICS_GRID = sphereicalGrid(
        az=np.linspace(0, 2 * np.pi, N_MICS, endpoint=False),
        co=np.full(N_MICS, np.pi / 2),
    )
    ARRAYS_TRAIN = [
        ArraySpec(
            name="rigid_sphere_r0.1_7mic",
            array_type="rigid_sphere",
            rigid_sphere=RigidSphereArrayConfig(
                mics_grid=RIGID_SPHERE_MICS_GRID, mic_radius=0.1
            ),
        ),
    ]
    ARRAYS_TEST = [
        ArraySpec(
            name="rigid_sphere_r0.1_7mic",
            array_type="rigid_sphere",
            rigid_sphere=RigidSphereArrayConfig(
                mics_grid=RIGID_SPHERE_MICS_GRID, mic_radius=0.1
            ),
        ),
        # To test generalisation to a new (unseen) array, add it here.
    ]
else:
    RIGID_SPHERE_MICS_GRID = None
    ARRAYS_TRAIN = []
    ARRAYS_TEST = []

# ============================================================
# === END USER CONFIG ========================================
# ============================================================


# ── path helpers ─────────────────────────────────────────────────────────────

def raw(*parts):
    return os.path.join(DATA_ROOT, "raw", *parts)


def prep(*parts):
    return os.path.join(DATA_ROOT, "prep", *parts)


def _get_ref_id(arr) -> int:
    """Return 0-based reference mic index for the given array.

    Tries REF_IDX_MAP first (for known pre-existing array names), then falls
    back to the mic geometrically closest to azimuth 0 (the target speaker
    direction after scene rotation).  Handles all three array types:
    rigid_sphere (uses mics_grid.az), free_field (derives az from Cartesian),
    precomputed (no positions available, defaults to 0).
    """
    key = arr.name + "_preprocessed"
    if key in REF_IDX_MAP:
        return REF_IDX_MAP[key] - 1  # map stores 1-based; convert to 0-based
    if arr.rigid_sphere is not None:
        return find_ref_mic(arr.rigid_sphere.mics_grid.az)
    if arr.free_field is not None:
        pos = arr.free_field.mic_positions
        az = np.arctan2(pos[:, 1], pos[:, 0])
        return find_ref_mic(az)
    return 0  # precomputed array — no mic positions available


# ============================================================
# Phase 1: Generate raw data
# ============================================================

def generate(mode, arrays_train, arrays_test):
    """Synthesise raw data from speech + room simulation."""

    if mode in ("ambidrop", "both"):
        print(f"  Type A (ideal SH) → {raw('ambidrop_train')} / {raw('ambidrop_val')}")
        gen_ambidrop(N_TRAIN, seed=0, output_root=raw("ambidrop_train"), speech_dir=SPEECH_TRAIN)
        gen_ambidrop(N_VAL,   seed=1, output_root=raw("ambidrop_val"),   speech_dir=SPEECH_VAL)

    if mode in ("baseline", "both"):
        print(f"  Type B (mic only) — {len(arrays_train)} training array(s)")
        gen_baseline(arrays_train, N_TRAIN, seed=0,
                     output_root=raw("baseline_train"), speech_dir=SPEECH_TRAIN)
        gen_baseline(arrays_train, N_VAL,   seed=1,
                     output_root=raw("baseline_val"),   speech_dir=SPEECH_VAL)

    # Type C is needed for ALL test modes (baseline + ambidrop)
    print(f"  Type C (full) — {len(arrays_test)} test array(s)")
    gen_test(arrays_test, N_TEST, seed=2, output_root=raw("test"), speech_dir=SPEECH_TEST)


# ============================================================
# Phase 2: Preprocess
# ============================================================

def preprocess(mode, arrays_train, arrays_test, test_only=False, test_raw_dir=None):
    """Convert raw data to .pt files.

    Baseline train/val: all ARRAYS_TRAIN merged into one flat directory via
    preprocess_dataset_multi. Each .pt stores array_name and ref_id.
    Baseline test: per-array so results can be compared separately.

    AmbiDrop training: real-ACN 2-tuples (no STFT, preprocess_sh_time).
    AmbiDrop test: NO preprocessing — MicToRealAmbisonicsDataset runs on-the-fly.

    When test_only=True, train/val preprocessing is skipped. test_raw_dir
    overrides the default raw("test") location for baseline test data.
    """
    _raw_test = test_raw_dir or raw("test")

    if mode in ("baseline", "both"):
        if not test_only:
            # ── merged multi-array train / val ────────────────────────────────
            for split, raw_root, train_flag in [
                ("baseline_train", raw("baseline_train"), True),
                ("baseline_val",   raw("baseline_val"),   True),
            ]:
                print(f"\n  Merging {len(arrays_train)} array(s) → {prep(split)}")
                array_configs = [
                    {"raw_dir": os.path.join(raw_root, arr.name),
                     "array_name": arr.name, "ref_id": _get_ref_id(arr)}
                    for arr in arrays_train
                ]
                preprocess_dataset_multi(
                    array_configs, prep(split),
                    preprocess_fn=preprocess_mic, train=train_flag,
                )

        # ── per-array baseline test ────────────────────────────────────────
        for array in arrays_test:
            print(f"\n  {array.name}/baseline_test")
            preprocess_dataset(
                os.path.join(_raw_test, array.name),
                prep(array.name, "baseline_test"),
                preprocess_fn=preprocess_mic,
                ref_id=_get_ref_id(array), array_name=array.name, train=False,
            )

    if mode in ("ambidrop", "both"):
        if not test_only:
            # ── AmbiDrop train / val (array-agnostic Type A) ──────────────────
            for split, src in [
                ("ambidrop_train", raw("ambidrop_train")),
                ("ambidrop_val",   raw("ambidrop_val")),
            ]:
                print(f"\n  {split}")
                preprocess_dataset(
                    src, prep(split),
                    preprocess_fn=preprocess_sh_time,
                    train=True,
                )
        # AmbiDrop test: no preprocessing — handled in _test_ambidrop via
        # MicToRealAmbisonicsDataset on raw Type-C data


# ============================================================
# Phase 3: Train
# ============================================================

def _build_model(mode):
    mic_num = N_MICS if mode == "baseline" else (2 + 1) ** 2  # 9 for order-2 SH
    return tasnet_model.TasNet(
        mic_num, NUM_SPK, ENC_DIM, FEATURE_DIM, CH_DIM,
        FS, WIN, LAYER, STACK, KERNEL, CAUSAL,
        mode=mode,
        drop_prob=DROP_PROB,
        max_drop=MAX_DROP,
        drop_probs=None,
        dropout_type=DROP_TYPE,
    )


def _make_solver_args(mode, save_folder):
    """Build the namespace that Solver.__init__ expects."""
    return argparse.Namespace(
        use_cuda=int(torch.cuda.is_available()),
        mode=mode,
        epochs=EPOCHS,
        half_lr=1,
        early_stop=1,
        max_norm=5.0,
        save_folder=save_folder,
        checkpoint=0,
        continue_from="",
        model_path="final.pth.tar",
        print_freq=10,
    )


def _train_one(mode, device):
    if mode == "baseline":
        # Single merged dir — all training arrays combined
        tr_dir = prep("baseline_train")
        cv_dir = prep("baseline_val")
    else:
        tr_dir = prep("ambidrop_train")
        cv_dir = prep("ambidrop_val")

    tr_dataset = SimDS_preprocessed(tr_dir, ".", mode=mode)
    cv_dataset = SimDS_preprocessed(cv_dir, ".", mode=mode)
    tr_loader  = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cv_loader  = DataLoader(cv_dataset, batch_size=1, shuffle=False)

    print(f"  [{mode}] {len(tr_dataset)} train / {len(cv_dataset)} val")

    model = _build_model(mode)
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {k:,}")

    optimizer   = torch.optim.Adam(model.parameters(), lr=LR)
    save_folder = os.path.join(CKPT_DIR, mode)
    os.makedirs(save_folder, exist_ok=True)

    solver = Solver({"tr_loader": tr_loader, "cv_loader": cv_loader},
                    model, optimizer, _make_solver_args(mode, save_folder))
    solver.train()
    print(f"  Checkpoint saved → {save_folder}/final.pth.tar")


def train(mode, device):
    for m in (["baseline", "ambidrop"] if mode == "both" else [mode]):
        _train_one(m, device)


# ============================================================
# Phase 4: Test
# ============================================================

def _build_steering_matrix(array_spec):
    """
    Extract positive-frequency steering matrix V (M, F_pos, Q) from the built
    array. Used by MicToRealAmbisonicsDataset for on-the-fly ASM at test time.
    """
    from shroom.geometry.sampling import sphereicalGrid  # noqa: lazy import
    from shroom.utils.grid_utils import from_fibonacci_grid  # noqa: lazy import
    source_grid = from_fibonacci_grid(SOURCE_GRID_POINTS)
    array_freq  = build_array(
        array_spec.array_type, source_grid, FS, ARRAY_DURATION_SEC,
        rigid_sphere=array_spec.rigid_sphere,
        free_field=array_spec.free_field,
        precomputed=array_spec.precomputed,
    )
    data  = np.asarray(array_freq.data)          # (M, Q, F_full)
    F_pos = data.shape[2] // 2 + 1
    V     = data[:, :, :F_pos].transpose(0, 2, 1)  # (M, F_pos, Q)
    return V, source_grid.co, source_grid.az


def _load_model(mode, checkpoint_path):
    model   = _build_model(mode)
    package = torch.load(checkpoint_path, map_location="cpu")
    state   = package.get("state_dict", package)
    model.load_state_dict(state)
    model.eval()
    return model


def _run_baseline_on_dir(model, data_dir, data_type, device):
    """Run baseline inference on one preprocessed directory and return (noisy, enh) SI-SDR arrays."""
    test_ds    = SimDS_preprocessed(data_dir, data_type, mode="baseline")
    testloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    si_sdrs_noisy, si_sdrs_enh = [], []
    for data in testloader:
        noisy, clean, ref_id, *_ = data
        noisy = noisy.to(device)
        clean = clean.to(device)

        with torch.no_grad():
            enhanced = model(noisy, ref_ids=ref_id)

        T         = min(enhanced.shape[-1], clean.shape[-1])
        enh       = enhanced[0, 0, :T]
        ref       = clean[0, ref_id[0], :T]
        noisy_ref = noisy[0, ref_id[0], :T]

        si_sdrs_noisy.append(si_snr(noisy_ref.unsqueeze(0), ref.unsqueeze(0)).item())
        si_sdrs_enh.append(si_snr(enh.unsqueeze(0), ref.unsqueeze(0)).item())

    return np.array(si_sdrs_noisy), np.array(si_sdrs_enh)


def _run_ambidrop_on_dir(model, raw_arr_dir, device):
    """Run AmbiDrop inference on a single raw array directory.

    Reads precomputed real-ACN Ambisonics (anmt_array) from anm.mat in each ex_N/
    subfolder, runs the model, and measures SI-SDR. Noisy SI-SDR uses mic channel 0
    vs clean mic channel 0 to match the computation used for RESULTS.md.
    """
    test_ds    = PrecomputedASMDataset(raw_arr_dir)
    testloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    si_sdrs_noisy, si_sdrs_enh = [], []
    for noisy_mic, clean_mic, anmt, clean_anm in testloader:
        anmt      = anmt.to(device)
        clean_anm = clean_anm.to(device)
        noisy_mic = noisy_mic.to(device)
        clean_mic = clean_mic.to(device)

        with torch.no_grad():
            enhanced = model(anmt)                     # (1, 1, T_out)

        # Enhanced SI-SDR: model output vs a00 direct-path
        T         = min(enhanced.shape[-1], clean_anm.shape[-1])
        enh       = enhanced[0, 0, :T]
        ref       = clean_anm[0, :T]

        # Noisy SI-SDR: first mic channel vs first clean mic channel
        T_mic     = min(noisy_mic.shape[-1], clean_mic.shape[-1])
        noisy_r   = noisy_mic[0, 0, :T_mic]
        clean_r   = clean_mic[0, 0, :T_mic]

        si_sdrs_noisy.append(si_snr(noisy_r.unsqueeze(0), clean_r.unsqueeze(0)).item())
        si_sdrs_enh.append(si_snr(enh.unsqueeze(0), ref.unsqueeze(0)).item())

    return np.array(si_sdrs_noisy), np.array(si_sdrs_enh)


def _test_baseline(model, array, device):
    """Evaluate baseline on preprocessed mic test data (new-format prep dirs)."""
    return _run_baseline_on_dir(model, prep(array.name, "baseline_test"), ".", device)


def _test_ambidrop(model, array, device, test_raw_dir=None):
    """Evaluate AmbiDrop on raw Type-C data via on-the-fly ASM (no preprocessing)."""
    _raw_test = test_raw_dir or raw("test")
    V, th, ph  = _build_steering_matrix(array)
    test_ds    = MicToRealAmbisonicsDataset(os.path.join(_raw_test, array.name), V, th, ph)
    testloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    si_sdrs_noisy, si_sdrs_enh = [], []
    for noisy_mic, clean_mic, anmt, clean_anm in testloader:
        anmt      = anmt.to(device)
        clean_anm = clean_anm.to(device)
        noisy_mic = noisy_mic.to(device)
        clean_mic = clean_mic.to(device)

        with torch.no_grad():
            enhanced = model(anmt)

        T         = min(enhanced.shape[-1], clean_anm.shape[-1])
        enh       = enhanced[0, 0, :T]
        ref       = clean_anm[0, :T]
        noisy_ref = noisy_mic[0, 0, :T]
        clean_ref = clean_mic[0, 0, :T]

        si_sdrs_noisy.append(si_snr(noisy_ref.unsqueeze(0), clean_ref.unsqueeze(0)).item())
        si_sdrs_enh.append(si_snr(enh.unsqueeze(0), ref.unsqueeze(0)).item())

    return np.array(si_sdrs_noisy), np.array(si_sdrs_enh)


def _print_si_sdr(mode, label, noisy, enh):
    print(f"\n  [{mode}] {label}")
    print(f"  Noisy    SI-SDR: {noisy.mean():.2f} dB")
    print(f"  Enhanced SI-SDR: {enh.mean():.2f} dB  (SI-SDRi: {(enh - noisy).mean():.2f} dB)")


def _test_one(mode, arrays_test, device, checkpoint=None, test_raw_dir=None,
              legacy_eval_dir=None):
    if checkpoint:
        ckpt = checkpoint
    else:
        # Priority: (1) newly trained model from _train_one → (2) preferred research checkpoint
        trained_ckpt   = os.path.join(CKPT_DIR, mode, "final.pth.tar")
        preferred_ckpt = os.path.join(CKPT_DIR,
                                      CKPT_BASELINE if mode == "baseline" else CKPT_AMBIDROP)
        ckpt = trained_ckpt if os.path.exists(trained_ckpt) else preferred_ckpt

    if not os.path.exists(ckpt):
        print(f"  [skip] No checkpoint at {ckpt}")
        return

    model = _load_model(mode, ckpt).to(device)

    # Determine whether to use legacy existing data or new-format prep dirs
    use_legacy = legacy_eval_dir or os.path.isdir(EVAL_TRAIN_DIR) or os.path.isdir(EVAL_TEST_DIR)

    if use_legacy:
        if mode == "baseline":
            eval_dirs = [legacy_eval_dir] if legacy_eval_dir else [EVAL_TRAIN_DIR, EVAL_TEST_DIR]
            for eval_dir in eval_dirs:
                if not os.path.isdir(eval_dir):
                    continue
                print(f"\n  [{mode}] evaluating → {eval_dir}")
                for array_name in sorted(
                    d for d in os.listdir(eval_dir)
                    if os.path.isdir(os.path.join(eval_dir, d)) and not d.startswith('.')
                ):
                    noisy_s, enh_s = _run_baseline_on_dir(model, eval_dir, array_name, device)
                    _print_si_sdr(mode, array_name, noisy_s, enh_s)
        else:  # ambidrop — uses raw dirs with precomputed anmt_array
            raw_dirs = [RAW_TRAIN_DIR, RAW_TEST_DIR]
            for raw_dir in raw_dirs:
                if not os.path.isdir(raw_dir):
                    continue
                print(f"\n  [{mode}] evaluating → {raw_dir}")
                for array_name in sorted(
                    d for d in os.listdir(raw_dir)
                    if os.path.isdir(os.path.join(raw_dir, d)) and not d.startswith('.')
                ):
                    arr_dir = os.path.join(raw_dir, array_name)
                    noisy_s, enh_s = _run_ambidrop_on_dir(model, arr_dir, device)
                    _print_si_sdr(mode, array_name, noisy_s, enh_s)
        return

    # New-format prep dirs
    for array in arrays_test:
        print(f"\n  [{mode}] {array.name}")
        if mode == "baseline":
            noisy_sisdrs, enh_sisdrs = _test_baseline(model, array, device)
        else:
            noisy_sisdrs, enh_sisdrs = _test_ambidrop(model, array, device, test_raw_dir)
        _print_si_sdr(mode, array.name, noisy_sisdrs, enh_sisdrs)


def test(mode, arrays_test, device, checkpoint=None,
         checkpoint_baseline=None, checkpoint_ambidrop=None, test_raw_dir=None,
         legacy_eval_dir=None):
    for m in (["baseline", "ambidrop"] if mode == "both" else [mode]):
        ckpt = (checkpoint_baseline if m == "baseline" else checkpoint_ambidrop) or checkpoint
        _test_one(m, arrays_test, device, ckpt, test_raw_dir, legacy_eval_dir=legacy_eval_dir)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="IC Conv-TasNet end-to-end run script")
    p.add_argument("--mode", choices=["baseline", "ambidrop", "both"], default="both",
                   help="Which pipeline to run")
    p.add_argument("--actions", nargs="+",
                   choices=["generate", "preprocess", "train", "test"],
                   default=["generate", "preprocess", "train", "test"],
                   help="Which phases to execute (default: all)")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint path for the test phase (single-mode runs)")
    p.add_argument("--checkpoint-baseline", default=None,
                   help="Checkpoint path for baseline model (used with --mode both)")
    p.add_argument("--checkpoint-ambidrop", default=None,
                   help="Checkpoint path for AmbiDrop model (used with --mode both)")
    p.add_argument("--test-raw-dir", default=None,
                   help="Override raw test data directory (default: DATA_ROOT/raw/test)")
    p.add_argument("--legacy-eval-dir", default=None,
                   help="Use a single existing preprocessed dir (baseline) or raw dir (ambidrop) "
                        "instead of the defaults.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device: {device}  |  mode: {args.mode}  |  actions: {args.actions}")
    print(f"Train arrays: {[a.name for a in ARRAYS_TRAIN]}")
    print(f"Test  arrays: {[a.name for a in ARRAYS_TEST]}")

    if "generate" in args.actions:
        print("\n=== Phase 1: Generate ===")
        generate(args.mode, ARRAYS_TRAIN, ARRAYS_TEST)

    if "preprocess" in args.actions:
        print("\n=== Phase 2: Preprocess ===")
        test_only = "train" not in args.actions and "generate" not in args.actions
        preprocess(args.mode, ARRAYS_TRAIN, ARRAYS_TEST,
                   test_only=test_only, test_raw_dir=args.test_raw_dir)

    if "train" in args.actions:
        print("\n=== Phase 3: Train ===")
        train(args.mode, device)

    if "test" in args.actions:
        print("\n=== Phase 4: Test ===")
        test(args.mode, ARRAYS_TEST, device,
             checkpoint=args.checkpoint,
             checkpoint_baseline=args.checkpoint_baseline,
             checkpoint_ambidrop=args.checkpoint_ambidrop,
             test_raw_dir=args.test_raw_dir,
             legacy_eval_dir=args.legacy_eval_dir)


if __name__ == "__main__":
    main()
