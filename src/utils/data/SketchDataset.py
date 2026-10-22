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

# ROOT = Path('data')
ROOT = Path('C:\\Users\\Utente\\Drake\\GENSKETCH\\data')
INDEX = ROOT / 'index.json'
SEED = 42
PHOTOS = 'photo'
SKETCHES = 'sketch'


def create_index(k: int = 5):
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

    val_fold = (test_fold + 1) % 5

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
    def __init__(self, root, records, size=256):
        self.root = root
        self.records = records
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        path = self.root / record['sketch']

        with Image.open(path) as image:
            image = image.convert('L').resize((self.size, self.size))
            array = np.array(image, dtype=np.float32)

        sketch = torch.from_numpy(array).unsqueeze(0) / 255.0
        return sketch, record['label']


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
    
#     selected = {'motorcycle'}
#     train_records, val_records = get_train_val_records(Path(ROOT / 'index.json'), selected)

#     train_dataset = SketchClassifierDataset(ROOT, train_records)
#     val_dataset = SketchClassifierDataset(ROOT, val_records)

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
