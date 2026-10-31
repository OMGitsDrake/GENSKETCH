"""Visualizza uno sketch caricato dal Dataset del progetto.

Esempi, dalla root del repository:
    py -m src.utils.preview_sketch --category cat
    py -m src.utils.preview_sketch --category cat --split train --fold 0 --augment
    py -m src.utils.preview_sketch --category cat --augment --output preview.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from .data.SketchDataset import SketchClassifierDataset, get_split_records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--category", help="Categoria da visualizzare; default: tutte")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--fold", type=int, choices=range(5), default=0,
                        help="Fold di test, usato quando --split non è all")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Indice dello sketch nei record selezionati")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--augment", action="store_true",
                        help="Mostra originale e variante RandomAffine affiancati")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed della trasformazione casuale")
    parser.add_argument("--output", type=Path,
                        help="Salva una PNG invece di aprire il visualizzatore")
    return parser.parse_args()


def load_tensor(dataset, index):
    # Subset seleziona un record; DataLoader aggiunge la dimensione batch.
    loader = DataLoader(Subset(dataset, [index]), batch_size=1, shuffle=False)
    sketches, labels = next(iter(loader))
    return sketches[0], int(labels[0].item())


def to_image(sketch):
    pixels = (sketch.squeeze(0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


def main():
    args = parse_args()
    root = args.data_root.resolve()
    index_path = root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Indice non trovato: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    categories = set(index["class_to_id"])
    if args.category is not None:
        if args.category not in categories:
            raise ValueError(f"Categoria sconosciuta: {args.category}")
        categories = {args.category}

    if args.split == "all":
        records = [r for r in index["records"] if r["category"] in categories]
    else:
        train, val, test = get_split_records(index_path, categories, args.fold)
        records = {"train": train, "val": val, "test": test}[args.split]

    if not records:
        raise ValueError("Nessuno sketch trovato con i filtri richiesti")
    if not 0 <= args.sample_index < len(records):
        raise IndexError(f"sample-index deve essere tra 0 e {len(records) - 1}")

    dataset = SketchClassifierDataset(root, records, size=args.size)
    sketch, label = load_tensor(dataset, args.sample_index)
    images = [("originale", to_image(sketch))]

    if args.augment:
        torch.manual_seed(args.seed)
        augmented_dataset = SketchClassifierDataset(root, records, size=args.size,
                                                     augment=True)
        augmented, augmented_label = load_tensor(augmented_dataset, args.sample_index)
        if augmented_label != label:
            raise RuntimeError("L'augmentation ha cambiato l'etichetta")
        images.append(("augmentato", to_image(augmented)))

    record = records[args.sample_index]
    if label != record["label"]:
        raise RuntimeError("L'etichetta del Dataset non coincide con index.json")

    banner_height = 24
    canvas = Image.new("RGB", (args.size * len(images), args.size + banner_height), "white")
    draw = ImageDraw.Draw(canvas)
    for position, (name, image) in enumerate(images):
        left = position * args.size
        canvas.paste(image, (left, banner_height))
        draw.text((left + 5, 5), name, fill="black")

    print(f"Categoria: {record['category']} (label {label})")
    print(f"Split: {args.split}; fold del record: {record['fold']}")
    print(f"Sketch: {root / record['sketch']}")
    print(f"Foto associata: {root / record['photo']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(args.output)
        print(f"Anteprima salvata in: {args.output.resolve()}")
    else:
        canvas.show()


if __name__ == "__main__":
    main()
