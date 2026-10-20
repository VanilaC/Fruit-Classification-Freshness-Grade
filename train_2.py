
import re
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

from Model_2 import FruitFreshnessModel


DATA_DIR = Path(
    r"C:\Users\PC\Desktop\MODEL\Fruit Model\Classification\label_data_class"
)
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = Path("checkpoints")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
FREEZE_EPOCHS = 5
LEARNING_RATE = 0.0001
VAL_RATIO = 0.15
SEED = 43
FRESHNESS_WEIGHT = 1.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_folder(folder_name):
    name = re.sub(r"[\s_-]+", " ", folder_name.strip().lower())

    match = re.match(
        r"^(semi fresh|semi rotten|fresh|mild|rotten)\s*(.+)$",
        name
    )

    if not match:
        return None

    condition, produce = match.groups()
    produce = re.sub(r"\s+", "", produce)

    if condition == "fresh":
        score = 1.0
    elif condition in {"mild", "semi fresh", "semi rotten"}:
        score = 0.5
    else:
        score = 0.0

    return produce, score


class ProduceDataset(Dataset):
    def __init__(self, root, transform, class_to_idx=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Folder not found: {self.root}")

        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue

            parsed = parse_folder(folder.name)
            if parsed is None:
                continue

            produce, score = parsed

            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, produce, score))

        if not self.samples:
            raise RuntimeError(f"No valid images found in {self.root}")

        discovered_classes = sorted({s[1] for s in self.samples})

        if class_to_idx is None:
            self.class_to_idx = {
                name: i for i, name in enumerate(discovered_classes)
            }
        else:
            self.class_to_idx = class_to_idx
            unknown = set(discovered_classes) - set(class_to_idx)
            if unknown:
                raise ValueError(f"Unknown test classes: {sorted(unknown)}")

        self.class_names = [
            name for name, _ in sorted(
                self.class_to_idx.items(), key=lambda item: item[1]
            )
        ]

        print(f"{self.root}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, produce, score = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return (
            image,
            torch.tensor(self.class_to_idx[produce], dtype=torch.long),
            torch.tensor(score, dtype=torch.float32)
        )


def evaluate(model, loader, cls_loss_fn, reg_loss_fn):
    model.eval()

    loss_total = 0.0
    correct = 0
    count = 0
    absolute_error = 0.0

    with torch.inference_mode():
        for images, labels, scores in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            scores = scores.to(DEVICE)

            logits, predictions = model(images)

            cls_loss = cls_loss_fn(logits, labels)
            reg_loss = reg_loss_fn(predictions, scores)
            loss = cls_loss + FRESHNESS_WEIGHT * reg_loss

            batch_size = images.size(0)
            loss_total += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            absolute_error += (predictions - scores).abs().sum().item()
            count += batch_size

    return {
        "loss": loss_total / count,
        "accuracy": 100.0 * correct / count,
        "mae": absolute_error / count
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Create a separate test folder first: {TEST_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    full_train = ProduceDataset(DATA_DIR, train_transform)
    class_names = full_train.class_names

    train_eval = ProduceDataset(
        DATA_DIR, eval_transform, full_train.class_to_idx
    )

    test_dataset = ProduceDataset(
        TEST_DIR, eval_transform, full_train.class_to_idx
    )

    if len(full_train) < 2:
        raise RuntimeError("Not enough images for training and validation.")

    val_size = max(1, int(len(full_train) * VAL_RATIO))
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(full_train), generator=generator
    ).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_loader = DataLoader(
        Subset(full_train, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    val_loader = DataLoader(
        Subset(train_eval, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

    model = FruitFreshnessModel(
        num_produce_classes=len(class_names),
        pretrained=True
    ).to(DEVICE)

    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    best_val_loss = float("inf")

    print(f"Device: {DEVICE}")
    print(f"Produce classes: {class_names}")
    print(f"Training: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Testing: {len(test_dataset)}")

    for epoch in range(NUM_EPOCHS):
        freeze_backbone = epoch < FREEZE_EPOCHS

        for param in model.features.parameters():
            param.requires_grad = not freeze_backbone

        model.train()

        train_loss_total = 0.0
        train_correct = 0
        train_count = 0

        for images, labels, scores in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            scores = scores.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits, predictions = model(images)

            cls_loss = cls_loss_fn(logits, labels)
            reg_loss = reg_loss_fn(predictions, scores)
            loss = cls_loss + FRESHNESS_WEIGHT * reg_loss

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss_total += loss.item() * batch_size
            train_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            train_count += batch_size

        train_loss = train_loss_total / train_count
        train_acc = 100.0 * train_correct / train_count

        val_metrics = evaluate(
            model, val_loader, cls_loss_fn, reg_loss_fn
        )

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Backbone: {'Frozen' if freeze_backbone else 'Unfrozen'} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val Freshness MAE: {val_metrics['mae']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": full_train.class_to_idx,
                "best_val_loss": best_val_loss
            }, OUTPUT_DIR / "best_model.pth")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "class_to_idx": full_train.class_to_idx,
            "best_val_loss": best_val_loss
        }, OUTPUT_DIR / "last_checkpoint.pth")

    checkpoint = torch.load(
        OUTPUT_DIR / "best_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model, test_loader, cls_loss_fn, reg_loss_fn
    )

    print("\nFINAL TEST RESULTS — BEST VALIDATION CHECKPOINT")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Produce classification accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Freshness score MAE: {test_metrics['mae']:.4f}")
    print(f"Models saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
