from .hmax import HMAX
from pathlib import Path


def load_hmax(**kwargs):
    # p = Path.cwd() / 'solo/backbones/hmax/universal_patch_set.mat'
    p = Path.cwd().parent / 'solo/backbones/hmax/universal_patch_set.mat'
    return HMAX(str(p))
