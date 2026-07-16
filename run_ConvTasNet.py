"""
End-to-end wrapper for the IC Conv-TasNet speech enhancement pipeline.

Orchestrates four sequential phases — generate, preprocess, train, test — for
both baseline (microphone input) and AmbiDrop (real-ACN Ambisonics) modes.
All data is preprocessed to .pt files before training or evaluation.

Public interface:
    generate   — synthesise Type A/B/C raw data from speech and room simulation
    preprocess — convert raw ex_N/ folders to .pt files
    train      — train Conv-TasNet model(s) via Solver and save checkpoints
    test       — evaluate trained model(s) and print SI-SDR per array

Phases (all enabled by default, skip any with --actions):
  generate    → synthesise raw data from speech + room simulation
  preprocess  → AmbiDrop train: real-ACN 2-tuples via preprocess_sh_time
                AmbiDrop test:  ASM-encoded 4-tuples via preprocess_ambisonics_time
                Baseline:       time-domain mic dicts via preprocess_mic_time
  train       → train Conv-TasNet model(s) via the Solver class
  test        → evaluate trained model(s) and print SI-SDR

Array configuration
  ARRAYS_TRAIN  — arrays merged into a single flat training/val dataset.
  ARRAYS_TEST   — arrays evaluated separately at test time.
                  Need not overlap with ARRAYS_TRAIN.

Usage:
    # First time — generate everything, then train and evaluate:
    python run_ConvTasNet.py --mode both --actions generate preprocess train test

    # Raw data already exists, skip generation:
    python run_ConvTasNet.py --mode ambidrop --actions preprocess train test

    # Use existing raw data from a different location (will preprocess it first):
    python run_ConvTasNet.py --mode both --actions preprocess train \
        --raw-ambidrop-train datasets/experiment_full_anm/raw/ambidrop_train \
        --raw-ambidrop-val   datasets/experiment_full_anm/raw/ambidrop_val \
        --raw-baseline-train datasets/experiment_full_anm/raw/baseline_train \
        --raw-baseline-val   datasets/experiment_full_anm/raw/baseline_val

    # Train directly from already-preprocessed .pt files (skip generate + preprocess):
    python run_ConvTasNet.py --mode ambidrop --actions train \
        --prep-ambidrop-train /path/to/preprocessed/train \
        --prep-ambidrop-val   /path/to/preprocessed/val

    # Evaluate from an existing checkpoint (no training):
    python run_ConvTasNet.py --mode ambidrop --actions test \
        --checkpoint checkpoints/run_convtasnet/ambidrop/final.pth.tar

    # Evaluate on both test AND train arrays (generates fresh data for both):
    python run_ConvTasNet.py --mode both --actions generate preprocess test --test-arrays both

    # Evaluate on train arrays only (skip test-array geometry):
    python run_ConvTasNet.py --mode both --actions generate preprocess test --test-arrays train

    # Evaluate on both using pre-existing raw dirs:
    python run_ConvTasNet.py --mode both --actions preprocess test \
        --test-raw-dir-test  datasets/.../raw/test \
        --test-raw-dir-train datasets/.../raw/train_eval
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
from ConvTasNet.datasets import SimDS_preprocessed
from ConvTasNet.preprocess import preprocess_mic_time, preprocess_ambisonics_time
from ambidrop.preprocess import (
    preprocess_sh_time,
    preprocess_dataset, preprocess_dataset_multi,
)
from ambidrop.losses import si_snr
from ambidrop.constants import get_device, REF_IDX_MAP, get_ref_idx
from ambidrop.signal_utils import find_ref_mic

try:
    from datagenerator.helpers import build_array, RigidSphereArrayConfig
    from datagenerator.generate_ambidrop_train_ds import generate_dataset as gen_ambidrop
    from datagenerator.generate_baseline_train_ds import (
        generate_dataset as gen_baseline,
        ArraySpec,
    )
    from datagenerator.generate_inference_ds import generate_dataset as gen_test
    from datagenerator.paper_arrays import PAPER_ARRAYS_TRAIN, PAPER_ARRAYS_TEST
    _DATAGENERATOR_AVAILABLE = True
except ImportError:
    _DATAGENERATOR_AVAILABLE = False
    ArraySpec = None
    RigidSphereArrayConfig = None
    PAPER_ARRAYS_TRAIN = []
    PAPER_ARRAYS_TEST  = []


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

# ── Preferred checkpoints for --actions test (inside CKPT_DIR) ───────────────
# After training via this script the newly trained model takes priority.
CKPT_BASELINE = "run_2026-04-09_10-55/final.pth.tar"
CKPT_AMBIDROP = "run_2026-04-09_08-35/final.pth.tar"

# ── Dataset sizes ─────────────────────────────────────────────────────────────
N_TRAIN_AMBIDROP = 200   # examples in AmbiDrop training split
N_VAL_AMBIDROP   = 40    # examples in AmbiDrop validation split
N_TRAIN_BASELINE = 200   # examples per array in baseline training split
N_VAL_BASELINE   = 40    # examples per array in baseline validation split
N_TEST           = 2    # examples per array in test split

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

DROP_TYPE  = "SHChannelDropout"
DROP_PROB  = 0.4
MAX_DROP   = 3
DROP_PROBS = None   # per-channel probs for PerChDropout, e.g. [0,0.1,0.45,...]; None → use DROP_PROB/MAX_DROP

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 8
LR         = 1e-3

# ── Array configurations ──────────────────────────────────────────────────────
# ARRAYS_TRAIN: merged into one flat training dataset.
# ARRAYS_TEST:  each array evaluated separately (can include unseen geometries).

N_MICS = 7
ARRAY_DURATION_SEC = 0.032  # used to build steering matrix for on-the-fly ASM

PAPER_ARRAYS = True  # True → use the 21 paper geometries; False → custom single-array setup

if _DATAGENERATOR_AVAILABLE:
    from shroom.geometry.sampling import sphereicalGrid  # noqa: lazy import

    if PAPER_ARRAYS:
        ARRAYS_TRAIN = PAPER_ARRAYS_TRAIN
        ARRAYS_TEST  = PAPER_ARRAYS_TEST
    else:
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
    ARRAYS_TRAIN = []
    ARRAYS_TEST  = []

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
    back to find_ref_mic which selects the mic closest to (r, 0, 0) in 3D.
    Handles all three array types:
    rigid_sphere (uses mics_grid.vecs scaled by mic_radius),
    free_field (uses mic_positions directly),
    precomputed (no positions available, defaults to 0).
    """
    if arr.name in REF_IDX_MAP:
        return REF_IDX_MAP[arr.name] - 1  # map stores 1-based; convert to 0-based
    if arr.rigid_sphere is not None:
        grid = arr.rigid_sphere.mics_grid
        pos = grid.vecs * arr.rigid_sphere.mic_radius  # (M, 3) in metres
        return find_ref_mic(pos)
    if arr.free_field is not None:
        return find_ref_mic(arr.free_field.mic_positions)
    return 0  # precomputed array — no mic positions available


# ============================================================
# Phase 1: Generate raw data
# ============================================================

def generate(mode, arrays_train, arrays_test, train_data=True, include_train_eval=False):
    """Synthesise raw data from speech + room simulation.

    train_data=False skips Type A and Type B (train/val splits) and only
    generates Type C (test data). Use this when training is not in the
    action list and you only need data for evaluation.
    include_train_eval=True also generates Type C data for train arrays
    (written to raw("test_train_arrays")) so they can be evaluated separately.
    """
    if train_data:
        if mode in ("ambidrop", "both"):
            print(f"  Type A (ideal SH) → {raw('ambidrop_train')} / {raw('ambidrop_val')}")
            gen_ambidrop(N_TRAIN_AMBIDROP, seed=0, output_root=raw("ambidrop_train"), speech_dir=SPEECH_TRAIN)
            gen_ambidrop(N_VAL_AMBIDROP,   seed=1, output_root=raw("ambidrop_val"),   speech_dir=SPEECH_VAL)

        if mode in ("baseline", "both"):
            print(f"  Type B (mic only) — {len(arrays_train)} training array(s)")
            gen_baseline(arrays_train, N_TRAIN_BASELINE, seed=0,
                         output_root=raw("baseline_train"), speech_dir=SPEECH_TRAIN)
            gen_baseline(arrays_train, N_VAL_BASELINE,   seed=1,
                         output_root=raw("baseline_val"),   speech_dir=SPEECH_VAL)

    if arrays_test:
        print(f"  Type C (full) — {len(arrays_test)} test array(s)")
        gen_test(arrays_test, N_TEST, seed=2, output_root=raw("test"), speech_dir=SPEECH_TEST)
    if include_train_eval:
        print(f"  Type C (full) — {len(arrays_train)} train array(s)  [for eval]")
        gen_test(arrays_train, N_TEST, seed=3, output_root=raw("test_train_arrays"), speech_dir=SPEECH_TEST)


# ============================================================
# Phase 2: Preprocess
# ============================================================

def preprocess(mode, arrays_train, arrays_test, test_only=False,
               test_raw_dir_test=None, test_raw_dir_train=None,
               raw_baseline_train=None, raw_baseline_val=None,
               raw_ambidrop_train=None, raw_ambidrop_val=None):
    """Convert raw data to .pt files.

    Baseline train/val: all ARRAYS_TRAIN merged into one flat directory via
    preprocess_dataset_multi. Each .pt stores array_name and ref_id.
    Baseline test: per-array so results can be compared separately.

    AmbiDrop training: real-ACN 2-tuples via preprocess_sh_time.
    AmbiDrop test: ASM-encoded 4-tuple dicts via preprocess_ambisonics_time.

    When test_only=True, train/val preprocessing is skipped.
    test_raw_dir_test overrides raw("test") for test-array eval data.
    test_raw_dir_train, when set, also preprocesses train arrays for evaluation.
    """
    _raw_test       = test_raw_dir_test or raw("test")
    _raw_train_eval = test_raw_dir_train  # None → skip train-array eval preprocessing

    if mode in ("baseline", "both"):
        if not test_only:
            # ── merged multi-array train / val ────────────────────────────────
            for split, raw_root, train_flag in [
                ("baseline_train", raw_baseline_train or raw("baseline_train"), True),
                ("baseline_val",   raw_baseline_val   or raw("baseline_val"),   True),
            ]:
                print(f"\n  Merging {len(arrays_train)} array(s) → {prep(split)}")
                array_configs = [
                    {"raw_dir": os.path.join(raw_root, arr.name),
                     "array_name": arr.name, "ref_id": _get_ref_id(arr)}
                    for arr in arrays_train
                ]
                preprocess_dataset_multi(
                    array_configs, prep(split),
                    preprocess_fn=preprocess_mic_time, train=train_flag,
                )

        # ── per-array eval: test arrays ───────────────────────────────────
        for array in arrays_test:
            print(f"\n  {array.name}/baseline_test")
            preprocess_dataset(
                os.path.join(_raw_test, array.name),
                prep(array.name, "baseline_test"),
                preprocess_fn=preprocess_mic_time,
                ref_id=_get_ref_id(array), array_name=array.name, train=False,
            )

        # ── per-array eval: train arrays (only when raw dir provided) ─────
        if _raw_train_eval:
            for array in arrays_train:
                print(f"\n  {array.name}/baseline_test  [train-array eval]")
                preprocess_dataset(
                    os.path.join(_raw_train_eval, array.name),
                    prep(array.name, "baseline_test"),
                    preprocess_fn=preprocess_mic_time,
                    ref_id=_get_ref_id(array), array_name=array.name, train=False,
                )

    if mode in ("ambidrop", "both"):
        if not test_only:
            # ── AmbiDrop train / val (array-agnostic Type A) ──────────────────
            for split, src in [
                ("ambidrop_train", raw_ambidrop_train or raw("ambidrop_train")),
                ("ambidrop_val",   raw_ambidrop_val   or raw("ambidrop_val")),
            ]:
                print(f"\n  {split}")
                preprocess_dataset(
                    src, prep(split),
                    preprocess_fn=preprocess_sh_time,
                    train=True,
                )

        # ── AmbiDrop test: encode ASM from raw p.wav per array ────────────────
        for array in arrays_test:
            print(f"\n  {array.name}/ambidrop_test")
            V, th, ph = _build_steering_matrix(array)
            preprocess_dataset(
                os.path.join(_raw_test, array.name),
                prep(array.name, "ambidrop_test"),
                preprocess_fn=preprocess_ambisonics_time,
                V=V, th=th, ph=ph,
                ref_id=_get_ref_id(array), array_name=array.name, train=False,
            )

        if _raw_train_eval:
            for array in arrays_train:
                print(f"\n  {array.name}/ambidrop_test  [train-array eval]")
                V, th, ph = _build_steering_matrix(array)
                preprocess_dataset(
                    os.path.join(_raw_train_eval, array.name),
                    prep(array.name, "ambidrop_test"),
                    preprocess_fn=preprocess_ambisonics_time,
                    V=V, th=th, ph=ph,
                    ref_id=_get_ref_id(array), array_name=array.name, train=False,
                )


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
        drop_probs=DROP_PROBS,
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


def _train_one(mode, device, prep_train=None, prep_val=None):
    if mode == "baseline":
        # Single merged dir — all training arrays combined
        tr_dir = prep_train or prep("baseline_train")
        cv_dir = prep_val   or prep("baseline_val")
    else:
        tr_dir = prep_train or prep("ambidrop_train")
        cv_dir = prep_val   or prep("ambidrop_val")

    tr_dataset = SimDS_preprocessed(tr_dir, ".", mode=mode)
    cv_dataset = SimDS_preprocessed(cv_dir, ".", mode=mode)
    tr_loader  = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)
    cv_loader  = DataLoader(cv_dataset, batch_size=1, shuffle=False)

    print(f"  [{mode}] {len(tr_dataset)} train / {len(cv_dataset)} val")
    if mode == "ambidrop":
        if DROP_PROBS is not None:
            print(f"  dropout: {DROP_TYPE}  drop_probs={DROP_PROBS}")
        else:
            print(f"  dropout: {DROP_TYPE}  prob={DROP_PROB}  max_drop={MAX_DROP}")

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


def train(mode, device, prep_baseline_train=None, prep_baseline_val=None,
          prep_ambidrop_train=None, prep_ambidrop_val=None):
    for m in (["baseline", "ambidrop"] if mode == "both" else [mode]):
        pt = prep_baseline_train if m == "baseline" else prep_ambidrop_train
        pv = prep_baseline_val   if m == "baseline" else prep_ambidrop_val
        _train_one(m, device, prep_train=pt, prep_val=pv)


# ============================================================
# Phase 4: Test
# ============================================================

def _build_steering_matrix(array_spec):
    """
    Extract positive-frequency steering matrix V (M, F_pos, Q) from the built
    array. Used by preprocess_ambisonics_time to encode ASM during preprocessing.
    """
    import scipy.io
    from shroom.geometry.sampling import sphereicalGrid  # noqa: lazy import
    grid_mat    = scipy.io.loadmat("utils/Lebvedev2702.mat")
    source_grid = sphereicalGrid(az=grid_mat["ph"].ravel(), co=grid_mat["th"].ravel())
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


def _test_baseline(model, array, device):
    """Evaluate baseline on preprocessed mic test data (new-format prep dirs)."""
    return _run_baseline_on_dir(model, prep(array.name, "baseline_test"), ".", device)


def _test_ambidrop(model, array, device):
    """Evaluate AmbiDrop on preprocessed ambidrop_test data."""
    test_ds    = SimDS_preprocessed(prep(array.name, "ambidrop_test"), '.', mode='ambidrop')
    testloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    si_sdrs_noisy, si_sdrs_enh = [], []
    for noisy_mic, clean_mic, anmt, clean_anm, ref_ids in testloader:
        ref_id    = int(ref_ids[0])
        anmt      = anmt.to(device)
        clean_anm = clean_anm.to(device)
        noisy_mic = noisy_mic.to(device)
        clean_mic = clean_mic.to(device)

        with torch.no_grad():
            enhanced = model(anmt)

        # Enhanced SI-SDR: model output vs a00 direct-path
        T         = min(enhanced.shape[-1], clean_anm.shape[-1])
        enh       = enhanced[0, 0, :T]
        ref       = clean_anm[0, :T]

        # Noisy SI-SDR: reference mic vs its clean direct-path
        noisy_ref = noisy_mic[0, ref_id, :T]
        clean_ref = clean_mic[0, ref_id, :T]

        si_sdrs_noisy.append(si_snr(noisy_ref.unsqueeze(0), clean_ref.unsqueeze(0)).item())
        si_sdrs_enh.append(si_snr(enh.unsqueeze(0), ref.unsqueeze(0)).item())

    return np.array(si_sdrs_noisy), np.array(si_sdrs_enh)


def _print_si_sdr(mode, label, noisy, enh):
    print(f"\n  [{mode}] {label}")
    print(f"  Noisy    SI-SDR: {noisy.mean():.2f} dB")
    print(f"  Enhanced SI-SDR: {enh.mean():.2f} dB  (SI-SDRi: {(enh - noisy).mean():.2f} dB)")


def _test_one(mode, arrays_test, device, checkpoint=None, force_fresh=False, arrays_train=None):
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

    for array in arrays_test:
        if mode == "baseline":
            noisy_sisdrs, enh_sisdrs = _test_baseline(model, array, device)
        else:
            noisy_sisdrs, enh_sisdrs = _test_ambidrop(model, array, device)
        _print_si_sdr(mode, array.name, noisy_sisdrs, enh_sisdrs)

    if arrays_train:
        for array in arrays_train:
            if mode == "baseline":
                noisy_sisdrs, enh_sisdrs = _test_baseline(model, array, device)
            else:
                noisy_sisdrs, enh_sisdrs = _test_ambidrop(model, array, device)
            _print_si_sdr(mode, f"{array.name} [train arr]", noisy_sisdrs, enh_sisdrs)


def test(mode, arrays_test, device, checkpoint=None,
         checkpoint_baseline=None, checkpoint_ambidrop=None,
         force_fresh=False, arrays_train=None):
    for m in (["baseline", "ambidrop"] if mode == "both" else [mode]):
        ckpt = (checkpoint_baseline if m == "baseline" else checkpoint_ambidrop) or checkpoint
        _test_one(m, arrays_test, device, ckpt,
                  force_fresh=force_fresh, arrays_train=arrays_train)


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
    p.add_argument("--test-arrays", choices=["test", "train", "both"], default="test",
                   help="Which array geometries to evaluate in the test phase: "
                        "'test' (default, ARRAYS_TEST only), 'train' (ARRAYS_TRAIN only), "
                        "or 'both' (all arrays)")
    p.add_argument("--test-raw-dir-test", default=None,
                   help="Raw Type-C data directory for test arrays "
                        "(default: DATA_ROOT/raw/test)")
    p.add_argument("--test-raw-dir-train", default=None,
                   help="Raw Type-C data directory for train arrays; "
                        "when set, also preprocesses and evaluates train arrays")
    p.add_argument("--raw-baseline-train", default=None,
                   help="Existing raw baseline training directory "
                        "(overrides DATA_ROOT/raw/baseline_train; skips generate for this split)")
    p.add_argument("--raw-baseline-val", default=None,
                   help="Existing raw baseline validation directory "
                        "(overrides DATA_ROOT/raw/baseline_val)")
    p.add_argument("--raw-ambidrop-train", default=None,
                   help="Existing raw AmbiDrop training directory "
                        "(overrides DATA_ROOT/raw/ambidrop_train; skips generate for this split)")
    p.add_argument("--raw-ambidrop-val", default=None,
                   help="Existing raw AmbiDrop validation directory "
                        "(overrides DATA_ROOT/raw/ambidrop_val)")
    p.add_argument("--prep-baseline-train", default=None,
                   help="Already-preprocessed baseline training directory "
                        "(used directly for training, skips preprocess step)")
    p.add_argument("--prep-baseline-val", default=None,
                   help="Already-preprocessed baseline validation directory")
    p.add_argument("--prep-ambidrop-train", default=None,
                   help="Already-preprocessed AmbiDrop training directory "
                        "(used directly for training, skips preprocess step)")
    p.add_argument("--prep-ambidrop-val", default=None,
                   help="Already-preprocessed AmbiDrop validation directory")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    print(f"Device: {device}  |  mode: {args.mode}  |  actions: {args.actions}")
    print(f"Train arrays: {[a.name for a in ARRAYS_TRAIN]}")
    print(f"Test  arrays: {[a.name for a in ARRAYS_TEST]}")

    test_train_eval   = args.test_arrays in ("train", "both")
    test_test_eval    = args.test_arrays in ("test",  "both")
    eval_test_arrays  = ARRAYS_TEST  if test_test_eval  else []
    eval_train_arrays = ARRAYS_TRAIN if test_train_eval else None

    if "generate" in args.actions:
        print("\n=== Phase 1: Generate ===")
        generate(args.mode, ARRAYS_TRAIN, eval_test_arrays,
                 train_data="train" in args.actions,
                 include_train_eval=test_train_eval)

    # Freshly generated train-eval data takes priority over the CLI arg;
    # fall back to the default location if raw data already exists on disk.
    if test_train_eval and "generate" in args.actions:
        effective_train_raw = raw("test_train_arrays")
    elif args.test_raw_dir_train:
        effective_train_raw = args.test_raw_dir_train
    elif test_train_eval and os.path.isdir(raw("test_train_arrays")):
        effective_train_raw = raw("test_train_arrays")
    else:
        effective_train_raw = None

    if "preprocess" in args.actions:
        print("\n=== Phase 2: Preprocess ===")
        preprocess(args.mode, ARRAYS_TRAIN, eval_test_arrays,
                   test_only="train" not in args.actions,
                   test_raw_dir_test=args.test_raw_dir_test,
                   test_raw_dir_train=effective_train_raw,
                   raw_baseline_train=args.raw_baseline_train,
                   raw_baseline_val=args.raw_baseline_val,
                   raw_ambidrop_train=args.raw_ambidrop_train,
                   raw_ambidrop_val=args.raw_ambidrop_val)

    if "train" in args.actions:
        print("\n=== Phase 3: Train ===")
        train(args.mode, device,
              prep_baseline_train=args.prep_baseline_train,
              prep_baseline_val=args.prep_baseline_val,
              prep_ambidrop_train=args.prep_ambidrop_train,
              prep_ambidrop_val=args.prep_ambidrop_val)

    if "test" in args.actions:
        print("\n=== Phase 4: Test ===")
        fresh = "generate" in args.actions or "preprocess" in args.actions
        eval_train = eval_train_arrays if effective_train_raw is not None else None
        test(args.mode, eval_test_arrays, device,
             checkpoint=args.checkpoint,
             checkpoint_baseline=args.checkpoint_baseline,
             checkpoint_ambidrop=args.checkpoint_ambidrop,
             force_fresh=fresh,
             arrays_train=eval_train)


if __name__ == "__main__":
    main()
