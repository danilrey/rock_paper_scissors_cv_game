import argparse
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.train import train_all, train_svm, train_custom_cnn, train_mobilenet, save_classes_mapping
from src.evaluate import evaluate_models
from src.realtime import run_realtime_demo


def main():
    parser = argparse.ArgumentParser(
        description="Rock-Paper-Scissors Computer Vision System: Training, Evaluation & Real-Time AI Game."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "demo", "game", "all"],
        default="evaluate",
        help="Pipeline mode to execute: 'train', 'evaluate', 'demo' (live inference), 'game' (play vs AI), or 'all'.",
    )
    parser.add_argument(
        "--model",
        choices=["all", "svm", "cnn", "mobilenet"],
        default="all",
        help="Target model for training or inference.",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Webcam device index for live demo/game (default: 0).",
    )

    args = parser.parse_args()

    if args.mode == "train":
        save_classes_mapping()
        if args.model == "svm":
            train_svm()
        elif args.model == "cnn":
            train_custom_cnn()
        elif args.model == "mobilenet":
            train_mobilenet()
        else:
            train_all()

    elif args.mode == "evaluate":
        evaluate_models()

    elif args.mode == "demo":
        run_realtime_demo(cam_index=args.cam, model_type=args.model if args.model != "all" else "mobilenet", mode="demo")

    elif args.mode == "game":
        run_realtime_demo(cam_index=args.cam, model_type=args.model if args.model != "all" else "mobilenet", mode="game")

    elif args.mode == "all":
        print("🚀 Executing Full Pipeline: Training + Evaluation...")
        train_all()
        evaluate_models()


if __name__ == "__main__":
    main()
