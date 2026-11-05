import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import random
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import joblib
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import (
    CLASSES,
    MOBILENET_MODEL_PATH,
    CNN_MODEL_PATH,
    SVM_MODEL_PATH,
    IMG_SIZE,
    IMG_SIZE_SVM,
)


class RPSPredictor:
    def __init__(self, model_type="mobilenet"):
        self.model_type = model_type
        if model_type == "mobilenet" and MOBILENET_MODEL_PATH.exists():
            print(f"Loading MobileNetV2 model from {MOBILENET_MODEL_PATH}...")
            self.model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
        elif model_type == "svm" and SVM_MODEL_PATH.exists():
            print(f"Loading SVM Baseline model from {SVM_MODEL_PATH}...")
            self.model = joblib.load(SVM_MODEL_PATH)
        elif model_type == "cnn" and CNN_MODEL_PATH.exists():
            print(f"Loading Custom CNN model from {CNN_MODEL_PATH}...")
            self.model = tf.keras.models.load_model(CNN_MODEL_PATH)
        elif MOBILENET_MODEL_PATH.exists():
            print(f"Loading MobileNetV2 model from {MOBILENET_MODEL_PATH}...")
            self.model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
            self.model_type = "mobilenet"
        elif CNN_MODEL_PATH.exists():
            print(f"Loading Custom CNN model from {CNN_MODEL_PATH}...")
            self.model = tf.keras.models.load_model(CNN_MODEL_PATH)
            self.model_type = "cnn"
        elif SVM_MODEL_PATH.exists():
            print(f"Loading SVM model from {SVM_MODEL_PATH}...")
            self.model = joblib.load(SVM_MODEL_PATH)
            self.model_type = "svm"
        else:
            raise FileNotFoundError("No trained model found! Please run training first.")

    def predict(self, crop_bgr):
        if self.model_type == "svm":
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, IMG_SIZE_SVM)
            feat = (resized.flatten().astype(np.float32) / 255.0).reshape(1, -1)
            probs = self.model.predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            return CLASSES[idx], float(probs[idx]), probs

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        tensor = resized.astype(np.float32)

        if self.model_type == "mobilenet":
            tensor = preprocess_input(tensor)
        else:
            tensor = tensor / 255.0

        batch = np.expand_dims(tensor, axis=0)
        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx]), probs


def determine_winner(player_move, computer_move):
    if player_move == computer_move:
        return "tie"
    wins = {
        ("rock", "scissors"),
        ("scissors", "paper"),
        ("paper", "rock"),
    }
    if (player_move, computer_move) in wins:
        return "player"
    return "computer"


def run_realtime_demo(cam_index=0, model_type="mobilenet", mode="demo"):
    predictor = RPSPredictor(model_type=model_type)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f" Could not open webcam at index {cam_index}.")
        print("Please check camera permissions in System Settings -> Privacy & Security -> Camera.")
        return

    print("=" * 60)
    print(f"🎮 RPS Real-Time Mode: {mode.upper()}")
    print("Controls:")
    print("  Q     : Quit application")
    print("  SPACE : Start Game Round (in game mode)")
    print("  R     : Reset Game Scores")
    print("=" * 60)

    scores = {"player": 0, "computer": 0, "tie": 0}
    countdown_active = False
    countdown_start = 0
    countdown_duration = 3.0
    last_result_text = "Press SPACE to start round"
    last_comp_move = None
    last_player_move = None
    result_display_timer = 0

    fps_history = []
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        t_curr = time.time()
        fps = 1.0 / max(t_curr - t_prev, 1e-5)
        t_prev = t_curr
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_class = "None"
        confidence = 0.0
        probabilities = [0.0, 0.0, 0.0]
        hand_box = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]

                margin = 40
                x_min = max(0, int(min(x_coords) * w) - margin)
                x_max = min(w, int(max(x_coords) * w) + margin)
                y_min = max(0, int(min(y_coords) * h) - margin)
                y_max = min(h, int(max(y_coords) * h) + margin)

                hand_box = (x_min, y_min, x_max, y_max)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 128), 2)

                crop = frame[y_min:y_max, x_min:x_max]
                if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                    detected_class, confidence, probabilities = predictor.predict(crop)

        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Model: {predictor.model_type.upper()}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if mode == "demo":
            color = (0, 255, 0) if confidence > 0.75 else (0, 165, 255)
            cv2.putText(frame, f"Gesture: {detected_class.upper()} ({confidence*100:.1f}%)",
                        (w - 420, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            for idx, c_name in enumerate(CLASSES):
                bar_y = 110 + idx * 25
                bar_len = int(probabilities[idx] * 120)
                cv2.putText(frame, f"{c_name[:4]}:", (20, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (80, bar_y), (80 + 120, bar_y + 15), (50, 50, 50), -1)
                cv2.rectangle(frame, (80, bar_y), (80 + bar_len, bar_y + 15), (0, 200, 255), -1)

        elif mode == "game":
            score_text = f"Player: {scores['player']}  |  AI: {scores['computer']}  |  Ties: {scores['tie']}"
            cv2.putText(frame, score_text, (w // 2 - 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if countdown_active:
                elapsed = time.time() - countdown_start
                remaining = countdown_duration - elapsed
                if remaining > 0:
                    cd_num = int(remaining) + 1
                    cv2.putText(frame, f"SHOW GESTURE IN: {cd_num}", (w // 2 - 180, h // 2),
                                cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 255), 3)
                else:
                    countdown_active = False
                    last_comp_move = random.choice(CLASSES)
                    last_player_move = detected_class

                    if last_player_move in CLASSES:
                        outcome = determine_winner(last_player_move, last_comp_move)
                        if outcome == "player":
                            scores["player"] += 1
                            last_result_text = "🎉 YOU WIN!"
                        elif outcome == "computer":
                            scores["computer"] += 1
                            last_result_text = "🤖 AI WINS!"
                        else:
                            scores["tie"] += 1
                            last_result_text = "🤝 TIE GAME!"
                    else:
                        last_result_text = "⚠️ No hand detected! Round void."

                    result_display_timer = time.time()

            if time.time() - result_display_timer < 4.0 and last_comp_move is not None:
                panel_y = h - 90
                cv2.rectangle(frame, (0, panel_y), (w, h), (15, 15, 15), -1)
                res_line = f"You: {str(last_player_move).upper()}  vs  AI: {last_comp_move.upper()}  -->  {last_result_text}"
                cv2.putText(frame, res_line, (30, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Press SPACE to challenge AI!", (w // 2 - 180, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("Rock Paper Scissors AI", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" ") and mode == "game" and not countdown_active:
            countdown_active = True
            countdown_start = time.time()
        elif key == ord("r") and mode == "game":
            scores = {"player": 0, "computer": 0, "tie": 0}
            last_result_text = "Scores reset!"
            result_display_timer = time.time()

    cap.release()
    cv2.destroyAllWindows()
