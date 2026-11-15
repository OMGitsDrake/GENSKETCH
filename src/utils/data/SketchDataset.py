'''
Definizione delle classi utilizzate per caricare i dati di classificatore e VAE, rispettivamente classi SketchClassifier e SketchVAE Dataset
'''

import random
import torch
import json
import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomAffine, InterpolationMode
from .. import config as cfg
from ..constants import DATA

ROOT = DATA
INDEX = ROOT / 'index.json'
SEED = cfg.BASE_SEED
PHOTOS = 'photo'
SKETCHES = 'sketch'


def create_index(k: int = cfg.K_FOLDS):
    categories = sorted(
        p.name for p in (ROOT / PHOTOS).iterdir() if p.is_dir()
    )
    class_to_id = {name : i for i, name in enumerate(categories)}
    records = []

    for category in categories:
        label = class_to_id[category]

        photos = sorted((ROOT / PHOTOS / category).glob('*.jpg'))
        photo_ids = [p.stem for p in photos]

        random.Random(SEED + label).shuffle(photo_ids)
        fold_by_photo = {
            photo_id: position % k
            for position, photo_id in enumerate(photo_ids)
        }

        sketches = sorted((ROOT / SKETCHES / category).glob('*.png'))
        for sketch in sketches:
            photo_id = sketch.stem.rsplit('-', 1)[0]
            photo_path = ROOT / PHOTOS / category / f'{photo_id}.jpg'

            if not photo_path.is_file():
                raise FileNotFoundError(f'Missing photo for sketch {sketch}')

            records.append({
                'sketch': sketch.relative_to(ROOT).as_posix(),
                'photo': photo_path.relative_to(ROOT).as_posix(),
                'photo_id': photo_id,
                'category': category,
                'label': label,
                'fold': fold_by_photo[photo_id]
            })

    index = {'class_to_id': class_to_id, 'records': records}
    INDEX.write_text(json.dumps(index, indent=2), encoding='utf-8')
    return index


def get_split_records(idx_path: Path, categories: set[str], test_fold: int):
    if idx_path.exists():
        index = json.loads(idx_path.read_text(encoding='utf-8'))
    else:
        index = create_index()

    val_fold = (test_fold + 1) % cfg.K_FOLDS

    selected = [
        r for r in index['records']
        if r['category'] in categories
    ]

    train_records = [
        r for r in selected if r['fold'] not in (val_fold, test_fold)
    ]
    val_records = [
        r for r in selected if r['fold'] == val_fold
    ]
    test_records = [
        r for r in selected if r['fold'] == test_fold
    ]

    return train_records, val_records, test_records


class SketchClassifierDataset(Dataset):
    def __init__(self, root, records, size=cfg.IMAGE_SIZE, augment=False):
        self.root = root
        self.records = records
        self.size = size

        augmentation = cfg.TRAIN_AUGMENTATION
        if augmentation["name"] != "RandomAffine":
            raise ValueError("Il Dataset supporta soltanto RandomAffine")
        self.augmentation = RandomAffine(
            degrees=augmentation["degrees"],
            translate=augmentation["translate"],
            scale=augmentation["scale"],
            interpolation=InterpolationMode[augmentation["interpolation"]],
            fill=augmentation["fill"],
        ) if augment else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        path = self.root / record['sketch']

        with Image.open(path) as image:
            image = image.convert('L').resize((self.size, self.size))
            if self.augmentation is not None:
                image = self.augmentation(image)
            array = np.array(image, dtype=np.float32)

        sketch = torch.from_numpy(array).unsqueeze(0) / 255.0
        return sketch, record['label']
