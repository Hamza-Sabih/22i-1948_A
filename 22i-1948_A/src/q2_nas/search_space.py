"""Deliverable 5 - searchable CNN architecture space.

An architecture is a JSON-serialisable dict:

{
  "resolution": 32,                 # variable input resolution        {16, 24, 32}
  "stem_channels": 24,              #                                  {16, 24, 32}
  "stem_kernel": 3,                 #                                  {3, 5}
  "stages": [                       # variable depth: 2-4 stages x 1-4 blocks
     {"block": "dws",               # block type   {conv, dws (depthwise separable), mbconv}
      "depth": 2,                   # blocks in the stage            {1, 2, 3, 4}
      "channels": 32,               # filters (stage-specific choices)
      "kernel": 5,                  # kernel size  {3, 5, 7}
      "expansion": 1,               # mbconv expansion ratio {2, 4, 6} (1 for conv/dws)
      "activation": "hardswish",    # {relu, relu6, hardswish, silu, gelu}
      "stride": 1}, ...],
  "head_channels": 256,             # 1x1 conv before pooling (0 = none) {0, 128, 256, 512}
  "dropout": 0.1
}

Search dimensions required by the assignment: variable depth (stages + blocks per stage),
variable filters/channels, kernel sizes, depthwise-separable blocks, activation
functions, input resolution.
"""
import random

RESOLUTIONS = [16, 24, 32]
STEM_CHANNELS = [16, 24, 32]
STEM_KERNELS = [3, 5]
NUM_STAGES = [2, 3, 4]
DEPTHS = [1, 2, 3, 4]
KERNELS = [3, 5, 7]
BLOCK_TYPES = ["conv", "dws", "mbconv"]
EXPANSIONS = [2, 4, 6]
ACTIVATIONS = ["relu", "relu6", "hardswish", "silu", "gelu"]
HEAD_CHANNELS = [0, 128, 256, 512]
DROPOUTS = [0.0, 0.1, 0.2]
# channel choices grow with stage index (stage 0 is the highest-resolution stage)
STAGE_CHANNELS = [[16, 24, 32, 48], [32, 48, 64, 96], [64, 96, 128, 160], [128, 160, 192, 256]]


def search_space_summary():
    n = len(RESOLUTIONS) * len(STEM_CHANNELS) * len(STEM_KERNELS) * len(HEAD_CHANNELS) * len(DROPOUTS)
    per_stage = [len(DEPTHS) * len(c) * len(KERNELS) * (2 + len(EXPANSIONS)) * len(ACTIVATIONS) for c in STAGE_CHANNELS]
    total = 0
    for s in NUM_STAGES:
        prod = 1
        for i in range(s):
            prod *= per_stage[i]
        total += prod
    return {"resolutions": RESOLUTIONS, "stem_channels": STEM_CHANNELS, "stem_kernels": STEM_KERNELS,
            "num_stages": NUM_STAGES, "depths": DEPTHS, "kernels": KERNELS, "block_types": BLOCK_TYPES,
            "expansions": EXPANSIONS, "activations": ACTIVATIONS, "head_channels": HEAD_CHANNELS,
            "dropouts": DROPOUTS, "stage_channels": STAGE_CHANNELS, "approx_num_architectures": n * total}


def _finalise(cfg):
    """Strides: stage 0 keeps resolution, every later stage halves it."""
    for i, st in enumerate(cfg["stages"]):
        st["stride"] = 1 if i == 0 else 2
        if st["block"] != "mbconv":
            st["expansion"] = 1
    return cfg


def sample_optuna(trial):
    """Sample an architecture with an Optuna trial (TPE models each choice)."""
    cfg = {"resolution": trial.suggest_categorical("resolution", RESOLUTIONS),
           "stem_channels": trial.suggest_categorical("stem_channels", STEM_CHANNELS),
           "stem_kernel": trial.suggest_categorical("stem_kernel", STEM_KERNELS),
           "stages": []}
    n_stages = trial.suggest_categorical("num_stages", NUM_STAGES)
    for i in range(n_stages):
        block = trial.suggest_categorical(f"s{i}_block", BLOCK_TYPES)
        cfg["stages"].append({
            "block": block,
            "depth": trial.suggest_categorical(f"s{i}_depth", DEPTHS),
            "channels": trial.suggest_categorical(f"s{i}_channels", STAGE_CHANNELS[i]),
            "kernel": trial.suggest_categorical(f"s{i}_kernel", KERNELS),
            "expansion": trial.suggest_categorical(f"s{i}_expansion", EXPANSIONS) if block == "mbconv" else 1,
            "activation": trial.suggest_categorical(f"s{i}_act", ACTIVATIONS),
        })
    cfg["head_channels"] = trial.suggest_categorical("head_channels", HEAD_CHANNELS)
    cfg["dropout"] = trial.suggest_categorical("dropout", DROPOUTS)
    return _finalise(cfg)


def sample_random(rng: random.Random = None):
    """Uniform random architecture (used for tests / random-search comparison)."""
    rng = rng or random.Random()
    cfg = {"resolution": rng.choice(RESOLUTIONS), "stem_channels": rng.choice(STEM_CHANNELS),
           "stem_kernel": rng.choice(STEM_KERNELS), "stages": []}
    for i in range(rng.choice(NUM_STAGES)):
        block = rng.choice(BLOCK_TYPES)
        cfg["stages"].append({"block": block, "depth": rng.choice(DEPTHS), "channels": rng.choice(STAGE_CHANNELS[i]),
                              "kernel": rng.choice(KERNELS),
                              "expansion": rng.choice(EXPANSIONS) if block == "mbconv" else 1,
                              "activation": rng.choice(ACTIVATIONS)})
    cfg["head_channels"] = rng.choice(HEAD_CHANNELS)
    cfg["dropout"] = rng.choice(DROPOUTS)
    return _finalise(cfg)


def arch_to_string(cfg):
    st = "|".join(f"{s['block']}-d{s['depth']}-c{s['channels']}-k{s['kernel']}-e{s['expansion']}-{s['activation']}"
                  for s in cfg["stages"])
    return f"r{cfg['resolution']}_stem{cfg['stem_channels']}k{cfg['stem_kernel']}_[{st}]_head{cfg['head_channels']}"
