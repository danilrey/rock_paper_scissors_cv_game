import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from a2_src.config import (
    A2_MODELS_DIR,
    A2_FIGURES_DIR,
    SOTA_MODEL_PATH,
    ABLATION_MODEL_PATH,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    RANDOM_SEED,
)
from a2_src.sota_model import build_sota_model
from a2_src.data_loader import get_dataloaders


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_f1 = f1_score(all_targets, all_preds, average="macro")
    return val_loss, val_acc, val_f1


def plot_curves(history, output_path, title_prefix="SOTA ConvNeXt"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title(f"{title_prefix}: Loss vs. Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "s--", label="Val Accuracy", color="#d62728", linewidth=2)
    axes[1].set_title(f"{title_prefix}: Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curve figure to: {output_path}")


def run_training(use_artaug: bool = True, epochs: int = NUM_EPOCHS):
    set_seed(RANDOM_SEED)
    print(f"Training: {'SOTA with ArtAug' if use_artaug else 'Ablation (Without ArtAug)'}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs} | Batch size: {BATCH_SIZE} | Learning rate: {LEARNING_RATE}")

    train_loader, val_loader, test_loader = get_dataloaders(use_artaug_train=use_artaug)

    model = build_sota_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = 0.0
    save_path = SOTA_MODEL_PATH if use_artaug else ABLATION_MODEL_PATH

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f}, F1: {tr_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "use_artaug": use_artaug,
                },
                save_path,
            )
            print(f"  --> Saved new best checkpoint to {save_path.name} (Val F1: {val_f1:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")

    hist_filename = "history_sota.json" if use_artaug else "history_ablation_no_artaug.json"
    hist_path = A2_MODELS_DIR / hist_filename
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    fig_filename = "training_curves_sota.png" if use_artaug else "training_curves_ablation.png"
    fig_path = A2_FIGURES_DIR / fig_filename
    prefix = "SOTA ConvNeXt + ArtAug" if use_artaug else "ConvNeXt (No ArtAug)"
    plot_curves(history, fig_path, title_prefix=prefix)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true", help="Train ablation baseline without ArtAug")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    args = parser.parse_args()

    run_training(use_artaug=not args.ablation, epochs=args.epochs)
