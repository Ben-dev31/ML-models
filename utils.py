
from pathlib import Path

def make_dirs(base):
    p = Path(base)
    imgs = p / 'images'
    masks = p / 'masks'
    imgs.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    return imgs, masks

