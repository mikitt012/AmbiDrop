"""
FT-JNF checkpoint registry mapping filename to model architecture configuration.

Public interface:
    CHECKPOINT_REGISTRY — dict mapping checkpoint filename to architecture config dict
                          (keys: mode, input_dim, hidden1, hidden2, dropout, drop_prob, max_drop, drop_probs)
"""
# ── Checkpoint registry ──────────────────────────────────────────────────────
# Maps each checkpoint filename to the model/dropout configuration it was
# trained with. This ensures you always know how to reconstruct the model
# architecture for a given checkpoint.

CHECKPOINT_REGISTRY = {
    # ── Baseline FT-JNF (microphone input, 14 channels, no dropout) ──────
    "FT_JNF,2025-11-30_14-41-59.pt": {
        "mode": "baseline", "input_dim": 14, "hidden1": 256, "hidden2": 128,
    },
    "FT_JNF,2025-12-01_09-21-58.pt": {
        "mode": "baseline", "input_dim": 14, "hidden1": 64, "hidden2": 64,
    },
    "FT_JNF,2025-12-29_14-41-04.pt": {
        "mode": "baseline", "input_dim": 14, "hidden1": 64, "hidden2": 64,
    },
    "FT_JNF,2026-03-25_13-37-42.pt": {
        "mode": "baseline", "input_dim": 14, "hidden1": 64, "hidden2": 64,
    }, # the preferred baseline checkpoint for the AmbiDrop paper experiments

    # ── AmbiDrop FT-JNF — SHChannelDropout ───────────────────────────────
    "SH_FT_JNF,2025-12-01_09-21-33.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 256, "hidden2": 128,
        "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3,
    },
    "SH_FT_JNF,2025-12-01_10-08-18.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3,
    },
    "SH_FT_JNF,2025-12-04_15-20-29.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.7, "max_drop": 7,
    },
    "SH_FT_JNF,2025-12-04_21-18-51.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.6, "max_drop": 7,
    },
    "SH_FT_JNF,2025-12-05_03-31-01.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.5, "max_drop": 6,
    },
    "SH_FT_JNF,2025-12-21_17-36-40.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 7,
    },
    "SH_FT_JNF,2025-12-22_14-53-29.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.3, "max_drop": 6,
    },
    "SH_FT_JNF,2025-12-23_05-06-54.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.7, "max_drop": 3,
    },
    "SH_FT_JNF,2025-12-23_13-38-21.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.0, "max_drop": 0,
    },
    "SH_FT_JNF,2026-03-09_14-23-53.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3,
    },

    # ── AmbiDrop FT-JNF — PerChDropout (threshold-based) ────────────────
    "SH_FT_JNF,2025-12-04_15-45-32.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -10.0,
        "drop_probs": [0, 0.55, 0.9, 0.6, 1, 1, 1, 1, 1],
    },
    "SH_FT_JNF,2025-12-04_21-44-11.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -8.2,
        "drop_probs": [0, 0.35, 0.75, 0.2, 1, 1, 1, 1, 1],
    },
    "SH_FT_JNF,2025-12-05_03-53-11.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -7.2,
        "drop_probs": [0, 0.15, 0.6, 0.2, 0.95, 1, 0.95, 1, 0.95],
    },
    "SH_FT_JNF,2025-12-05_09-45-29.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -5.7,
        "drop_probs": [0, 0.15, 0.55, 0.15, 0.8, 1, 0.95, 1, 0.85],
    },
    "SH_FT_JNF,2025-12-05_15-44-31.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -5.0,
        "drop_probs": [0, 0.1, 0.45, 0.15, 0.7, 1, 0.85, 1, 0.65],
    },
    "SH_FT_JNF,2025-12-05_21-56-06.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -4.2,
        "drop_probs": [0, 0.1, 0.45, 0.1, 0.55, 1, 0.85, 1, 0.55],
    },
    "SH_FT_JNF,2025-12-06_04-07-37.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -3.4,
        "drop_probs": [0, 0.1, 0.45, 0.1, 0.45, 1, 0.75, 1, 0.45],
    },
    "SH_FT_JNF,2025-12-06_09-56-40.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -2.4,
        "drop_probs": [0, 0.05, 0.45, 0.05, 0.4, 0.95, 0.5, 0.95, 0.4],
    },
    "SH_FT_JNF,2025-12-06_15-58-32.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": -1.4,
        "drop_probs": [0, 0.05, 0.45, 0.05, 0.1, 0.75, 0.4, 0.75, 0.1],
    },
    "SH_FT_JNF,2025-12-06_22-04-29.pt": {
        "mode": "ambidrop", "input_dim": 18, "hidden1": 64, "hidden2": 64,
        "dropout": "PerChDropout", "threshold_dB": 0.0,
        "drop_probs": [0, 0.05, 0.35, 0.05, 0, 0.4, 0, 0.3, 0],
    },

    # ── Network size ablation (all SHChannelDropout p=0.4 max=3) ─────────
    "checkpoint_size_3490.pt":    {"mode": "ambidrop", "input_dim": 18, "hidden1": 8,   "hidden2": 8,   "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_7330.pt":    {"mode": "ambidrop", "input_dim": 18, "hidden1": 16,  "hidden2": 8,   "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_11074.pt":   {"mode": "ambidrop", "input_dim": 18, "hidden1": 16,  "hidden2": 16,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_23874.pt":   {"mode": "ambidrop", "input_dim": 18, "hidden1": 32,  "hidden2": 16,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_38530.pt":   {"mode": "ambidrop", "input_dim": 18, "hidden1": 32,  "hidden2": 32,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_84610.pt":   {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 32,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_142594.pt":  {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_316674.pt":  {"mode": "ambidrop", "input_dim": 18, "hidden1": 128, "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_547330.pt":  {"mode": "ambidrop", "input_dim": 18, "hidden1": 128, "hidden2": 128, "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "checkpoint_size_1223170.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 256, "hidden2": 128, "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},

    # ── Named experiment checkpoints ─────────────────────────────────────
    "AmbiDrop_full_experiment.pt":          {"mode": "ambidrop", "input_dim": 18, "hidden1": 256, "hidden2": 128, "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "AmbiDrop_full_experiment_smallnet.pt": {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "AmbiDrop_experiment2.pt":              {"mode": "ambidrop", "input_dim": 18, "hidden1": 64,  "hidden2": 64,  "dropout": "SHChannelDropout", "drop_prob": 0.4, "max_drop": 3},
    "baseline_experiment2.pt":              {"mode": "baseline", "input_dim": 14, "hidden1": 64,  "hidden2": 64},
}
