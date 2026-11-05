import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from a2_src.config import (
    BASE_DIR,
    A2_MODELS_DIR,
    A2_REPORTS_DIR,
    A2_FIGURES_DIR,
    SOTA_MODEL_PATH,
    ABLATION_MODEL_PATH,
    TEST_DIR,
    CLASSES,
    NUM_CLASSES,
    DEVICE,
)
from a2_src.sota_model import build_sota_model
from a2_src.data_loader import get_dataloaders, get_stress_test_loader
from a2_src.artaug import polish_low_exposure_frame, generate_stress_environment, get_eval_transforms
from a2_src.gradcam import GradCAM


def load_model(checkpoint_path: Path):
    model = build_sota_model(num_classes=NUM_CLASSES, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def evaluate_loader(model, loader, device=DEVICE):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weight),
        "targets": all_targets,
        "preds": all_preds,
    }


def benchmark_latency(model, device=DEVICE, num_runs=200):
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model.eval()

    for _ in range(30):
        with torch.no_grad():
            _ = model(dummy_input)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    fps = float(1000.0 / mean_ms)
    return mean_ms, std_ms, fps


def plot_confusion_matrix(targets, preds, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Ground Truth Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_gradcam_gallery(model, output_path):
    gradcam = GradCAM(model, model.get_gradcam_target_layer())
    eval_transform = get_eval_transforms()

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))

    for idx, c in enumerate(CLASSES):
        class_folder = TEST_DIR / c
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
        if not images:
            continue
        sample_path = images[0]
        bgr = cv2.imread(str(sample_path))
        bgr_resized = cv2.resize(bgr, (224, 224))
        rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)

        tensor_in = eval_transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
        heatmap = gradcam.generate_heatmap(tensor_in, target_class=idx)
        overlay = gradcam.overlay_heatmap(rgb, heatmap, alpha=0.55, colormap=cv2.COLORMAP_VIRIDIS)

        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f"Ground Truth: {c.capitalize()}\nOriginal Frame", fontsize=11)
        axes[idx, 0].axis("off")

        im = axes[idx, 1].imshow(heatmap, cmap="viridis")
        axes[idx, 1].set_title(f"Grad-CAM Heatmap\n(Activation Field)", fontsize=11)
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f"Visual Explanation\nDiscriminative Regions", fontsize=11, fontweight="bold")
        axes[idx, 2].axis("off")

    gradcam.remove_hooks()
    plt.suptitle("SOTA ConvNeXt Explainability via Grad-CAM", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM explainability panel to: {output_path}")


def run_full_evaluation():
    print("Assignment 2 evaluation")
    print(f"Device: {DEVICE}")

    if not SOTA_MODEL_PATH.exists():
        print(f"Error: {SOTA_MODEL_PATH} not found. Please run training first.")
        return

    sota_model = load_model(SOTA_MODEL_PATH)
    _, _, clean_test_loader = get_dataloaders()

    sota_clean = evaluate_loader(sota_model, clean_test_loader)
    mean_ms, std_ms, fps = benchmark_latency(sota_model)
    sota_clean["latency_ms"] = mean_ms
    sota_clean["fps"] = fps

    plot_confusion_matrix(
        sota_clean["targets"],
        sota_clean["preds"],
        A2_FIGURES_DIR / "confusion_matrix_sota.png",
        title="Confusion Matrix: SOTA ConvNeXt-Tiny (Clean Test)",
    )

    generate_gradcam_gallery(sota_model, A2_FIGURES_DIR / "gradcam_gallery.png")

    stress_scenarios = {
        "Clean Baseline": {"exposure": 1.0, "blur_kernel": 1, "noise_level": 0.0, "polish": False},
        "Low Light (0.35x)": {"exposure": 0.35, "blur_kernel": 1, "noise_level": 0.0, "polish": False},
        "Low Light + ArtAug Polish": {"exposure": 0.35, "blur_kernel": 1, "noise_level": 0.0, "polish": True},
        "Severe Motion Blur": {"exposure": 1.0, "blur_kernel": 15, "noise_level": 0.0, "polish": False},
        "Sensor Noise (ISO 6400)": {"exposure": 1.0, "blur_kernel": 1, "noise_level": 30.0, "polish": False},
        "Compound Stress (Dim+Blur+Noise)": {"exposure": 0.4, "blur_kernel": 11, "noise_level": 25.0, "polish": False},
    }

    stress_results_sota = {}
    for name, params in stress_scenarios.items():
        loader = get_stress_test_loader(
            exposure=params["exposure"],
            blur_kernel=params["blur_kernel"],
            noise_level=params["noise_level"],
            polish_frames=params["polish"],
        )
        res = evaluate_loader(sota_model, loader)
        stress_results_sota[name] = res["accuracy"]
        print(f"Scenario [{name}]: Accuracy = {res['accuracy']*100:.2f}% | F1 = {res['macro_f1']:.4f}")

    ablation_stress = {}
    if ABLATION_MODEL_PATH.exists():
        ablation_model = load_model(ABLATION_MODEL_PATH)
        for name, params in stress_scenarios.items():
            loader = get_stress_test_loader(
                exposure=params["exposure"],
                blur_kernel=params["blur_kernel"],
                noise_level=params["noise_level"],
                polish_frames=False,
            )
            res = evaluate_loader(ablation_model, loader)
            ablation_stress[name] = res["accuracy"]

    plt.figure(figsize=(11, 6))
    scenarios = list(stress_scenarios.keys())
    sota_accs = [stress_results_sota[s] * 100 for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.35

    plt.bar(x - width/2, sota_accs, width, label="SOTA (ConvNeXt + ArtAug)", color="#1f77b4")
    if ablation_stress:
        abl_accs = [ablation_stress[s] * 100 for s in scenarios]
        plt.bar(x + width/2, abl_accs, width, label="Ablation (Without ArtAug)", color="#ff7f0e")

    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title("Stress-Test Robustness Across Real-World Environmental Degeneracies", fontsize=13, fontweight="bold")
    plt.xticks(x, scenarios, rotation=20, ha="right", fontsize=10)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(A2_FIGURES_DIR / "stress_test_robustness.png", dpi=300)
    plt.close()

    a1_summary_path = BASE_DIR / "reports" / "metrics_summary.json"
    a1_data = {}
    if a1_summary_path.exists():
        with open(a1_summary_path) as f:
            a1_data = json.load(f)

    rows = []
    a1_models = a1_data.get("models", [])
    if isinstance(a1_data, dict) and not a1_models:
        a1_models = [
            {
                "name": name,
                "metrics": {
                    "accuracy": values.get("Accuracy"),
                    "macro_f1": values.get("Macro F1"),
                    "latency_ms": values.get("Latency (ms/frame)"),
                    "fps": values.get("FPS"),
                },
                "model_size_mb": values.get("Model Size (MB)"),
            }
            for name, values in a1_data.items()
            if isinstance(values, dict) and "Accuracy" in values
        ]

    for m in a1_models:
        metrics = m["metrics"]
        rows.append({
            "Stage": "Assignment 1 Baseline",
            "Model": m["name"],
            "Clean Accuracy": f"{metrics['accuracy']*100:.2f}%",
            "Macro F1": f"{metrics['macro_f1']:.4f}",
            "Latency (ms)": f"{metrics['latency_ms']:.2f}",
            "FPS": f"{metrics['fps']:.1f}",
            "Model Size (MB)": f"{m['model_size_mb']:.2f}",
        })

    rows.append({
        "Stage": "Assignment 2 SOTA",
        "Model": "SOTA ConvNeXt-Tiny + ArtAug",
        "Clean Accuracy": f"{sota_clean['accuracy']*100:.2f}%",
        "Macro F1": f"{sota_clean['macro_f1']:.4f}",
        "Latency (ms)": f"{sota_clean['latency_ms']:.2f}",
        "FPS": f"{sota_clean['fps']:.1f}",
        "Model Size (MB)": f"{SOTA_MODEL_PATH.stat().st_size / (1024 * 1024):.2f}",
    })

    df = pd.DataFrame(rows)
    df.to_csv(A2_REPORTS_DIR / "benchmark_table.csv", index=False)

    print("\nBenchmark Table generated successfully:")
    print(df.to_string(index=False))

    summary_data = {
        "sota_clean_metrics": {k: v for k, v in sota_clean.items() if not isinstance(v, np.ndarray)},
        "stress_test_accuracy": stress_results_sota,
        "ablation_stress_accuracy": ablation_stress,
    }
    with open(A2_REPORTS_DIR / "a2_evaluation_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)


if __name__ == "__main__":
    run_full_evaluation()
