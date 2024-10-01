from nni.compression.pruning import (
    LevelPruner,
    L1NormPruner,
    L2NormPruner,
    FPGMPruner,
    SlimPruner,
    TaylorPruner,
    MovementPruner,
    LinearPruner,
    AGPPruner
)

PRUNER_DICT = {
    'level': LevelPruner,
    'l1': L1NormPruner,
    'l2': L2NormPruner,
    'fpgm': FPGMPruner
}
