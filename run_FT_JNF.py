"""
End-to-end wrapper for the FT-JNF speech enhancement pipeline.

Orchestrates four sequential phases — generate, preprocess, train, test — for
both baseline (microphone input) and AmbiDrop (Ambisonics + SHChannelDropout) modes.

Public interface:
    generate — synthesise Type A/B/C raw data from speech and room simulation
    preprocess — convert raw ex_N/ folders to preprocessed .pt files
    train — train FT-JNF model(s) and save checkpoints
    test — evaluate trained model(s) and print SI-SDR / PESQ / STOI per array

Phases (all enabled by default, skip any with --actions):
  generate    → synthesise raw data from speech + room simulation
  preprocess  → convert raw data to .pt files with STFT tensors
  train       → train FT-JNF model(s)
  test        → evaluate trained model(s) and print SI-SDR / PESQ / STOI

Array configuration
  ARRAYS_TRAIN  — arrays used to generate training and validation data.
                  All arrays are merged into a single flat dataset so the
                  model trains on every geometry simultaneously.
  ARRAYS_TEST   — arrays used to generate and evaluate test data.
                  Each array is evaluated separately so you can compare results.
                  Need not overlap with ARRAYS_TRAIN (use unseen arrays to test
                  generalisation).

Usage:
    # First time — generate everything, then train and evaluate:
    python run_FT_JNF.py --mode both --actions generate preprocess train test

    # Raw data already exists, skip generation:
    python run_FT_JNF.py --mode baseline --actions preprocess train test

    # Use existing raw data from a different location (will preprocess it first):
    python run_FT_JNF.py --mode both --actions preprocess train \
        --raw-ambidrop-train datasets/experiment_full_anm/raw/ambidrop_train \
        --raw-ambidrop-val   datasets/experiment_full_anm/raw/ambidrop_val \
        --raw-baseline-train datasets/experiment_full_anm/raw/baseline_train \
        --raw-baseline-val   datasets/experiment_full_anm/raw/baseline_val

    # Train directly from already-preprocessed .pt files (skip generate + preprocess):
    python run_FT_JNF.py --mode ambidrop --actions train \
        --prep-ambidrop-train /path/to/preprocessed/train \
        --prep-ambidrop-val   /path/to/preprocessed/val

    # Evaluate from an existing checkpoint (no training):
    python run_FT_JNF.py --mode ambidrop --actions test \
        --checkpoint checkpoints/run_ftjnf/ambidrop.pt

    # Preprocess and evaluate only (no generate, no train):
    python run_FT_JNF.py --mode both --actions preprocess test \
        --checkpoint checkpoints/run_ftjnf/ambidrop.pt

    # Evaluate on both test AND train arrays (generates fresh data for both):
    python run_FT_JNF.py --mode both --actions generate preprocess test --test-arrays both

    # Evaluate on train arrays only (skip test-array geometry):
    python run_FT_JNF.py --mode both --actions generate preprocess test --test-arrays train

    # Evaluate on both using pre-existing raw dirs (paper Table I style):
    python run_FT_JNF.py --mode both --actions preprocess test \
        --test-raw-dir-test  datasets/.../raw/test \
        --test-raw-dir-train datasets/.../raw/train_eval

    # Generate new data, only for testing (skip train/val splits):
    python run_FT_JNF.py --mode both --actions generate preprocess test

"""

import os
import sys
import argparse
from typing import NamedTuple, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader

from shroom.geometry.sampling import sphereicalGrid

from FT_JNF.model import FT_JNF
from FT_JNF.datasets import SimDS_preprocessed
from FT_JNF.train import training_step_baseline, training_step_ambidrop
from FT_JNF.test_simulated import evaluate_array
from ambidrop.preprocess import (
    preprocess_mic, preprocess_sh_stft,
    preprocess_dataset, preprocess_dataset_multi,
)
from ambidrop.checkpoint import save_checkpoint, load_checkpoint
from ambidrop.constants import get_device, REF_IDX_MAP, get_ref_idx
from ambidrop.signal_utils import find_ref_mic
from datagenerator.helpers import RigidSphereArrayConfig
from datagenerator.generate_ambidrop_train_ds import generate_dataset as gen_ambidrop
from datagenerator.generate_baseline_train_ds import (
    generate_dataset as gen_baseline,
    ArraySpec,
)
from datagenerator.generate_inference_ds import generate_dataset as gen_test
from datagenerator.paper_arrays import PAPER_ARRAYS_TRAIN, PAPER_ARRAYS_TEST


class WandbCfg(NamedTuple):
    project:  str
    entity:   str
    run_name: Optional[str]
    enabled:  bool


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
DATA_ROOT = "datasets/run_ftjnf"       # raw + preprocessed data go here
CKPT_DIR  = "checkpoints/FT_JNF"      # newly trained checkpoints saved here

# ── Pre-existing evaluation datasets (generated by the full experiment) ───────
# Used by default when --actions test is run without generating new data.
# Set to None to force evaluation on freshly generated data instead.
EVAL_TRAIN_DIR = "datasets/experiment_full_anm/test_of_train_ds_preprocessed"
EVAL_TEST_DIR  = "datasets/experiment_full_anm/test_of_test_ds_preprocessed"

# ── Preferred checkpoints for --actions test (inside CKPT_DIR) ───────────────
# After training via this script the newly trained model takes priority
# automatically. These are fallbacks when no trained model is found.
CKPT_BASELINE = "FT_JNF,2026-03-25_13-37-42.pt"
CKPT_AMBIDROP = "SH_FT_JNF,2025-12-01_10-08-18.pt"

# ── Dataset sizes ─────────────────────────────────────────────────────────────
N_TRAIN_AMBIDROP = 1000   # examples in AmbiDrop training split
N_VAL_AMBIDROP   = 150    # examples in AmbiDrop validation split
N_TRAIN_BASELINE = 100   # examples per array in baseline training split
N_VAL_BASELINE   = 20    # examples per array in baseline validation split
N_TEST           = 2     # examples per array in test split

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS       = 100
BATCH_SIZE   = 8
LR           = 1e-3
WEIGHT_DECAY = 1e-6

# ── Model architecture ────────────────────────────────────────────────────────
INPUT_DIM_BASELINE = 14   # 7 mics × 2 (real + imag STFT)
INPUT_DIM_AMBIDROP = 18   # 9 SH channels × 2
HIDDEN1 = 64
HIDDEN2 = 64

DROP_TYPE  = "SHChannelDropout"
DROP_PROB  = 0.4
MAX_DROP   = 3
DROP_PROBS = None   # per-channel probs for PerChDropout, e.g. [0,0.1,0.45,...]; None → use DROP_PROB/MAX_DROP

# ── Array configurations ──────────────────────────────────────────────────────
# ARRAYS_TRAIN: merged into one flat training dataset.
# ARRAYS_TEST:  each array evaluated separately (can include unseen geometries).
# Add more ArraySpec entries here to train/test on additional array types.

PAPER_ARRAYS = True # choose between paper arrays or custom arrays

if PAPER_ARRAYS:
    ARRAYS_TRAIN = PAPER_ARRAYS_TRAIN
    ARRAYS_TEST  = PAPER_ARRAYS_TEST
else:
    N_MICS = 7

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
        # Add more training arrays here, e.g.:
        # ArraySpec(name="free_field_7mic", array_type="free_field",
        #           free_field=FreeFieldArrayConfig(mic_positions=...)),
    ]

    ARRAYS_TEST = [
        ArraySpec(
            name="rigid_sphere_r0.1_7mic",
            array_type="rigid_sphere",
            rigid_sphere=RigidSphereArrayConfig(
                mics_grid=RIGID_SPHERE_MICS_GRID, mic_radius=0.1
            ),
        ),
        # To test generalisation to a new (unseen) array, add it here without
        # adding it to ARRAYS_TRAIN. AmbiDrop should still generalise.
    ]


# ============================================================
# === END USER CONFIG ========================================
# ============================================================


# ── path helpers ─────────────────────────────────────────────────────────────

def raw(*parts):
    return os.path.join(DATA_ROOT, "raw", *parts)


def prep(*parts):
    return os.path.join(DATA_ROOT, "prep", *parts)


def _get_ref_id(arr: ArraySpec) -> int:
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
# Phase 2: Preprocess raw data → .pt files
# ============================================================

def preprocess(mode, arrays_train, arrays_test, test_only=False,
               test_raw_dir_test=None, test_raw_dir_train=None,
               raw_baseline_train=None, raw_baseline_val=None,
               raw_ambidrop_train=None, raw_ambidrop_val=None):
    """Convert raw data to .pt files with STFT tensors.

    Baseline train/val: all ARRAYS_TRAIN are merged into a single flat directory
    (ex_1.pt, ex_2.pt, …) via preprocess_dataset_multi. Each .pt stores
    array_name and ref_id so the training step picks the correct reference mic.

    Test: each array in ARRAYS_TEST (and optionally ARRAYS_TRAIN) gets its own
    prep directory so results can be reported per array.

    When test_only=True, train/val preprocessing is skipped (useful when raw
    train data doesn't exist but you want to evaluate an existing checkpoint).
    test_raw_dir_test overrides the default raw("test") location for test arrays.
    test_raw_dir_train, when set, also preprocesses train arrays for evaluation.
    raw_baseline_train/val and raw_ambidrop_train/val override the default
    raw("baseline_train") etc. paths so existing raw data can be used without
    running the generate phase first.
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
                    preprocess_fn=preprocess_mic, train=train_flag,
                )

        # ── per-array eval: test arrays ───────────────────────────────────
        for array in arrays_test:
            print(f"\n  {array.name}/baseline_test")
            preprocess_dataset(
                os.path.join(_raw_test, array.name),
                prep(array.name, "baseline_test"),
                preprocess_fn=preprocess_mic,
                ref_id=_get_ref_id(array), array_name=array.name, train=False,
            )

        # ── per-array eval: train arrays (only when raw dir provided) ─────
        if _raw_train_eval:
            for array in arrays_train:
                print(f"\n  {array.name}/baseline_test  [train-array eval]")
                preprocess_dataset(
                    os.path.join(_raw_train_eval, array.name),
                    prep(array.name, "baseline_test"),
                    preprocess_fn=preprocess_mic,
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
                    preprocess_fn=preprocess_sh_stft,
                    anm_source="ideal", train=True,
                )

        # ── per-array eval: test arrays (Type C, ASM-encoded) ────────────
        for array in arrays_test:
            print(f"\n  {array.name}/ambidrop_test")
            preprocess_dataset(
                os.path.join(_raw_test, array.name),
                prep(array.name, "ambidrop_test"),
                preprocess_fn=preprocess_sh_stft,
                anm_source="asm", ref_id=_get_ref_id(array), array_name=array.name, train=False,
            )

        # ── per-array eval: train arrays (only when raw dir provided) ─────
        if _raw_train_eval:
            for array in arrays_train:
                print(f"\n  {array.name}/ambidrop_test  [train-array eval]")
                preprocess_dataset(
                    os.path.join(_raw_train_eval, array.name),
                    prep(array.name, "ambidrop_test"),
                    preprocess_fn=preprocess_sh_stft,
                    anm_source="asm", ref_id=_get_ref_id(array), array_name=array.name, train=False,
                )


# ============================================================
# Phase 3: Train
# ============================================================

def _build_net(mode, device):
    if mode == "baseline":
        return FT_JNF(
            input_dim=INPUT_DIM_BASELINE,
            hidden1_dim=HIDDEN1, hidden2_dim=HIDDEN2, output_dim=2,
        ).to(device)
    return FT_JNF(
        input_dim=INPUT_DIM_AMBIDROP,
        hidden1_dim=HIDDEN1, hidden2_dim=HIDDEN2, output_dim=2,
        dropout_type=DROP_TYPE, drop_prob=DROP_PROB, max_drop=MAX_DROP,
        drop_probs=DROP_PROBS,
    ).to(device)


def _train_one(mode, device, wb: WandbCfg = None, prep_train=None, prep_val=None):
    net = _build_net(mode, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    step_fn = training_step_baseline if mode == "baseline" else training_step_ambidrop

    if mode == "baseline":
        # Single merged dir — all training arrays combined
        train_ds = SimDS_preprocessed(prep_train or prep("baseline_train"))
        val_ds   = SimDS_preprocessed(prep_val   or prep("baseline_val"))
        ckpt     = os.path.join(CKPT_DIR, "baseline.pt")
    else:
        train_ds = SimDS_preprocessed(prep_train or prep("ambidrop_train"))
        val_ds   = SimDS_preprocessed(prep_val   or prep("ambidrop_val"))
        ckpt     = os.path.join(CKPT_DIR, "ambidrop.pt")

    os.makedirs(CKPT_DIR, exist_ok=True)
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valloader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print(f"  [{mode}] {len(train_ds)} train / {len(val_ds)} val  →  {ckpt}")
    if mode == "ambidrop":
        if DROP_PROBS is not None:
            print(f"  dropout: {DROP_TYPE}  drop_probs={DROP_PROBS}")
        else:
            print(f"  dropout: {DROP_TYPE}  prob={DROP_PROB}  max_drop={MAX_DROP}")

    if wb and wb.enabled:
        run_name = f"{wb.run_name}_train_{mode}" if wb.run_name else f"train_{mode}"
        wandb.init(
            project=wb.project, entity=wb.entity or None,
            name=run_name, reinit=True,
            config={
                "mode": mode, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "lr": LR, "weight_decay": WEIGHT_DECAY,
                "hidden1": HIDDEN1, "hidden2": HIDDEN2,
                "arrays_train": [a.name for a in ARRAYS_TRAIN],
                **({"drop_type": DROP_TYPE, "drop_prob": DROP_PROB,
                    "max_drop": MAX_DROP, "drop_probs": DROP_PROBS}
                   if mode == "ambidrop" else {}),
            },
        )

    import time
    prev_loss = float("inf")
    for epoch in range(EPOCHS):
        t0 = time.time()
        net.train()
        train_loss = 0.0
        for data in trainloader:
            optimizer.zero_grad()
            loss = step_fn(net, data, device)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        net.eval()
        with torch.no_grad():
            val_loss = sum(step_fn(net, data, device).item() for data in valloader)

        total_train = train_loss / max(len(trainloader), 1)
        total_val   = val_loss   / max(len(valloader),   1)
        elapsed     = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}: train={total_train:.4f}  val={total_val:.4f}  ({elapsed:.1f}s)")

        if wb and wb.enabled:
            wandb.log({"train/loss": total_train, "val/loss": total_val, "epoch": epoch + 1})

        if total_val < prev_loss:
            prev_loss = total_val
            save_checkpoint(ckpt, epoch, net, optimizer, total_val)
            print(f"           ↳ saved checkpoint")
            if wb and wb.enabled:
                wandb.log({"val/best_loss": total_val, "best_epoch": epoch + 1})

    if wb and wb.enabled:
        wandb.finish()


def train(mode, device, wb: WandbCfg = None, prep_baseline_train=None, prep_baseline_val=None,
          prep_ambidrop_train=None, prep_ambidrop_val=None):
    for m in (["baseline", "ambidrop"] if mode == "both" else [mode]):
        pt = prep_baseline_train if m == "baseline" else prep_ambidrop_train
        pv = prep_baseline_val   if m == "baseline" else prep_ambidrop_val
        _train_one(m, device, wb=wb, prep_train=pt, prep_val=pv)


# ============================================================
# Phase 4: Test
# ============================================================

def _print_metrics(mode, label, metrics, log_to_wandb=False):
    print(f"\n  [{mode}] {label}")
    print(f"  Noisy    SI-SDR: {metrics['si_sdr_noisy'].mean():.2f} dB  "
          f"PESQ: {metrics['pesq_noisy'].mean():.2f}  "
          f"STOI: {metrics['stoi_noisy'].mean():.3f}")
    print(f"  Enhanced SI-SDR: {metrics['si_sdr_enhanced'].mean():.2f} dB  "
          f"PESQ: {metrics['pesq_enhanced'].mean():.2f}  "
          f"STOI: {metrics['stoi_enhanced'].mean():.3f}")
    if log_to_wandb:
        prefix = label.replace(" ", "_").replace("/", "-")
        wandb.log({
            f"{prefix}/si_sdr_noisy":    float(metrics['si_sdr_noisy'].mean()),
            f"{prefix}/si_sdr_enhanced": float(metrics['si_sdr_enhanced'].mean()),
            f"{prefix}/pesq_noisy":      float(metrics['pesq_noisy'].mean()),
            f"{prefix}/pesq_enhanced":   float(metrics['pesq_enhanced'].mean()),
            f"{prefix}/stoi_noisy":      float(metrics['stoi_noisy'].mean()),
            f"{prefix}/stoi_enhanced":   float(metrics['stoi_enhanced'].mean()),
        })


def _eval_legacy_dir(net, mode, device, eval_dir, log_to_wandb=False):
    """Evaluate all array subdirs inside an existing preprocessed directory."""
    array_types = sorted(
        d for d in os.listdir(eval_dir)
        if os.path.isdir(os.path.join(eval_dir, d)) and not d.startswith('.')
    )
    for test_type in array_types:
        metrics = evaluate_array(net, test_type, data_dir=eval_dir, mode=mode, device=device)
        _print_metrics(mode, test_type, metrics, log_to_wandb=log_to_wandb)


def _test_one(mode, arrays_test, device, checkpoint=None, legacy_eval_dir=None,
              force_fresh=False, arrays_train=None, wb: WandbCfg = None):
    net = _build_net(mode, device)

    if checkpoint:
        ckpt = checkpoint
    else:
        # Priority: (1) newly trained model from _train_one → (2) preferred research checkpoint
        trained_name  = "baseline.pt" if mode == "baseline" else "ambidrop.pt"
        preferred_name = CKPT_BASELINE if mode == "baseline" else CKPT_AMBIDROP
        trained_ckpt   = os.path.join(CKPT_DIR, trained_name)
        preferred_ckpt = os.path.join(CKPT_DIR, preferred_name)
        ckpt = trained_ckpt if os.path.exists(trained_ckpt) else preferred_ckpt

    if not os.path.exists(ckpt):
        print(f"  [skip] No checkpoint at {ckpt}")
        return

    load_checkpoint(ckpt, net=net)
    net.eval()

    log_wb = wb and wb.enabled
    if log_wb:
        run_name = f"{wb.run_name}_test_{mode}_" if wb.run_name else f"test_{mode}"
        wandb.init(
            project=wb.project, entity=wb.entity or None,
            name=run_name, reinit=True,
            config={
                "mode": mode, "checkpoint": ckpt,
                "arrays_test": [a.name for a in arrays_test],
            },
        )

    # Determine evaluation directories:
    #   explicit override → per-array prep dirs (when fresh data was generated)
    #   → RESULTS.md legacy dirs → per-array prep dirs as final fallback
    if legacy_eval_dir:
        for eval_dir in ([legacy_eval_dir] if isinstance(legacy_eval_dir, str) else legacy_eval_dir):
            print(f"\n  [{mode}] evaluating → {eval_dir}")
            _eval_legacy_dir(net, mode, device, eval_dir, log_to_wandb=log_wb)
        if log_wb:
            wandb.finish()
        return

    test_type = "baseline_test" if mode == "baseline" else "ambidrop_test"

    if force_fresh or not (os.path.isdir(EVAL_TRAIN_DIR) or os.path.isdir(EVAL_TEST_DIR)):
        # Use per-array prep directories (freshly generated or no legacy dirs available)
        for array in arrays_test:
            metrics = evaluate_array(
                net, test_type,
                data_dir=prep(array.name),
                mode=mode, device=device,
                ref_id=_get_ref_id(array),
                array_name=array.name,
            )
            _print_metrics(mode, array.name, metrics, log_to_wandb=log_wb)
        if arrays_train:
            for array in arrays_train:
                metrics = evaluate_array(
                    net, test_type,
                    data_dir=prep(array.name),
                    mode=mode, device=device,
                    ref_id=_get_ref_id(array),
                    array_name=array.name,
                )
                _print_metrics(mode, f"{array.name} [train arr]", metrics, log_to_wandb=log_wb)
        if log_wb:
            wandb.finish()
        return

    # Default: legacy RESULTS.md eval dirs
    for eval_dir in [d for d in [EVAL_TRAIN_DIR, EVAL_TEST_DIR] if os.path.isdir(d)]:
        print(f"\n  [{mode}] evaluating → {eval_dir}")
        _eval_legacy_dir(net, mode, device, eval_dir, log_to_wandb=log_wb)
    if arrays_train:
        for array in arrays_train:
            metrics = evaluate_array(
                net, test_type,
                data_dir=prep(array.name),
                mode=mode, device=device,
                ref_id=_get_ref_id(array),
                array_name=array.name,
            )
            _print_metrics(mode, f"{array.name} [train arr]", metrics, log_to_wandb=log_wb)

    if log_wb:
        wandb.finish()


def test(mode, arrays_test, device, checkpoint=None,
         checkpoint_baseline=None, checkpoint_ambidrop=None, legacy_eval_dir=None,
         force_fresh=False, arrays_train=None, wb: WandbCfg = None):
    for m in (["baseline", "ambidrop"] if mode == "both" else [mode]):
        ckpt = (checkpoint_baseline if m == "baseline" else checkpoint_ambidrop) or checkpoint
        _test_one(m, arrays_test, device, ckpt, legacy_eval_dir=legacy_eval_dir,
                  force_fresh=force_fresh, arrays_train=arrays_train, wb=wb)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="FT-JNF end-to-end run script")
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
    p.add_argument("--test-arrays", choices=["test", "train", "both"], default="both",
                   help="Which array geometries to generate/evaluate in the test phase: "
                        "'test' (default, ARRAYS_TEST only), 'train' (ARRAYS_TRAIN only), "
                        "or 'both' (all arrays)")
    p.add_argument("--test-raw-dir-test", default=None,
                   help="Raw Type-C data directory for test arrays "
                        "(default: DATA_ROOT/raw/test)")
    p.add_argument("--test-raw-dir-train", default=None,
                   help="Raw Type-C data directory for train arrays; "
                        "when set, also preprocesses and evaluates train arrays")
    p.add_argument("--legacy-eval-dir", default=None,
                   help="Use existing preprocessed dir directly (4-tuple format, "
                        "e.g. datasets/experiment_full_anm/test_of_train_ds_preprocessed). "
                        "Skips preprocessing; each subdir is treated as a separate array.")
    p.add_argument("--wandb-project", default="FT_JNF",
                   help="W&B project name (default: FT_JNF)")
    p.add_argument("--wandb-entity", default="",
                   help="W&B entity / team (default: your default entity)")
    p.add_argument("--wandb-run-name", default=None,
                   help="Optional prefix for W&B run names "
                        "(runs are named '{prefix}_train_{mode}' / '{prefix}_test_{mode}')")
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
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging entirely")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}  |  mode: {args.mode}  |  actions: {args.actions}")
    print(f"Train arrays: {[a.name for a in ARRAYS_TRAIN]}")
    print(f"Test  arrays: {[a.name for a in ARRAYS_TEST]}")

    wb = WandbCfg(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        enabled=not args.no_wandb,
    )

    test_train_eval = args.test_arrays in ("train", "both")
    test_test_eval  = args.test_arrays in ("test",  "both")
    eval_test_arrays = ARRAYS_TEST  if test_test_eval  else []
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
        train(args.mode, device, wb=wb,
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
             legacy_eval_dir=args.legacy_eval_dir,
             force_fresh=fresh,
             arrays_train=eval_train,
             wb=wb)


if __name__ == "__main__":
    main()
