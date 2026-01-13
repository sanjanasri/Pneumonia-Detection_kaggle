import os
import argparse
import random
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Base data dir that contains train/ and (optionally) val/ and test/")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of training images to move into validation set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    val_dir.mkdir(parents=True, exist_ok=True)

    # For each class create val/class_name folder
    for class_name in os.listdir(train_dir):
        class_train_dir = train_dir / class_name
        if not class_train_dir.is_dir():
            continue

        class_val_dir = val_dir / class_name
        class_val_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in os.listdir(class_train_dir) if (class_train_dir / f).is_file()]
        random.shuffle(images)

        n_val = int(len(images) * args.val_fraction)
        val_images = images[:n_val]

        print(f"Class {class_name}: moving {n_val} images to val/")

        for img_name in val_images:
            src = class_train_dir / img_name
            dst = class_val_dir / img_name
            shutil.move(src, dst)


if __name__ == "__main__":
    main()

