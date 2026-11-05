import sys
import random
import threading
import time
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
import av
from PIL import Image

import torch
import torch.nn.functional as F
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from a2_src.config import (
    SOTA_MODEL_PATH,
    CLASSES,
    NUM_CLASSES,
    DEVICE,
)
from a2_src.sota_model import build_sota_model
from a2_src.artaug import get_eval_transforms


@st.cache_resource
def get_prediction_state():
    return {
        "label": None,
        "confidence": 0.0,
        "updated_at": 0.0,
        "lock": threading.Lock(),
    }


prediction_state = get_prediction_state()


@st.cache_resource
def get_hand_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
    )


def crop_hand_region(frame_bgr):
    detector = get_hand_detector()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return frame_bgr, False

    height, width = frame_bgr.shape[:2]
    landmarks = results.multi_hand_landmarks[0].landmark
    margin = max(20, int(min(height, width) * 0.08))
    x_min = max(0, int(min(point.x for point in landmarks) * width) - margin)
    x_max = min(width, int(max(point.x for point in landmarks) * width) + margin)
    y_min = max(0, int(min(point.y for point in landmarks) * height) - margin)
    y_max = min(height, int(max(point.y for point in landmarks) * height) + margin)
    crop = frame_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return frame_bgr, False
    return crop, True


class GestureVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.transforms = eval_transforms
        self.prediction_state = prediction_state
        self.detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def recv(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        height, width = frame_bgr.shape[:2]
        result = self.detector.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0].landmark
            margin = max(20, int(min(height, width) * 0.08))
            x_min = max(0, int(min(point.x for point in landmarks) * width) - margin)
            x_max = min(width, int(max(point.x for point in landmarks) * width) + margin)
            y_min = max(0, int(min(point.y for point in landmarks) * height) - margin)
            y_max = min(height, int(max(point.y for point in landmarks) * height) + margin)
            hand_crop = frame_bgr[y_min:y_max, x_min:x_max]

            if hand_crop.size and hand_crop.shape[0] >= 20 and hand_crop.shape[1] >= 20:
                resized = cv2.resize(hand_crop, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = self.transforms(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probabilities = F.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probabilities))
                label = CLASSES[pred_idx].capitalize()
                confidence = probabilities[pred_idx] * 100.0
                with self.prediction_state["lock"]:
                    self.prediction_state["label"] = CLASSES[pred_idx]
                    self.prediction_state["confidence"] = confidence
                    self.prediction_state["updated_at"] = time.monotonic()

                cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 220, 100), 2)
                cv2.putText(
                    frame_bgr,
                    f"{label}  {confidence:.1f}%",
                    (x_min, max(30, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 220, 100),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame_bgr,
                    "Hand detected",
                    (x_min, min(height - 12, y_max + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 100),
                    2,
                    cv2.LINE_AA,
                )
                return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

        cv2.putText(
            frame_bgr,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")


st.set_page_config(
    page_title="SOTA Hand Gesture CV Demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


if not SOTA_MODEL_PATH.exists():
    st.error(
        f"Model checkpoint was not found: {SOTA_MODEL_PATH}. "
        "Train the A2 model from A2_Complete_Flow.ipynb first."
    )
    st.stop()


@st.cache_resource
def get_model():
    model = build_sota_model(num_classes=NUM_CLASSES, pretrained=False)
    if SOTA_MODEL_PATH.exists():
        ckpt = torch.load(SOTA_MODEL_PATH, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


model = get_model()
eval_transforms = get_eval_transforms()

st.title("Hand Gesture Recognition with SOTA CV")
st.markdown("**Hand Gesture Recognition** | Rock, Paper, Scissors  \n"
            "**SOTA method:** ConvNeXt-Tiny with ArtAug low-light enhancement")

st.divider()

st.sidebar.header("Live Camera Demo")
st.sidebar.caption("Start the stream and show a hand gesture to see the model prediction.")

st.subheader("Live camera")
st.write("The green box marks the hand region used by the model. The label above it is the current prediction.")
webrtc_streamer(
    key="rps-live-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=GestureVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if "game_round" not in st.session_state:
    st.session_state.game_round = 0
    st.session_state.user_score = 0
    st.session_state.computer_score = 0
    st.session_state.game_history = []


def get_round_winner(user_move, computer_move):
    if user_move == computer_move:
        return "Draw"
    winning_moves = {
        "rock": "scissors",
        "scissors": "paper",
        "paper": "rock",
    }
    return "You win" if winning_moves[user_move] == computer_move else "Computer wins"


st.subheader("Rock, Paper, Scissors")
with prediction_state["lock"]:
    current_move = prediction_state["label"]
    current_confidence = prediction_state["confidence"]

game_over = (
    st.session_state.game_round >= 3
    or st.session_state.user_score >= 2
    or st.session_state.computer_score >= 2
)
score_user, score_round, score_computer = st.columns(3)
score_user.metric("You", st.session_state.user_score)
score_round.metric("Round", f"{st.session_state.game_round}/3")
score_computer.metric("Computer", st.session_state.computer_score)

if current_move:
    st.write(f"Current gesture: **{current_move.capitalize()}** ({current_confidence:.1f}%)")
else:
    st.write("Current gesture: show your hand in the camera.")

play_round, new_game = st.columns(2)
play_clicked = play_round.button("Play round", type="primary", use_container_width=True, disabled=game_over)
new_game_clicked = new_game.button("New game", use_container_width=True)

if new_game_clicked:
    st.session_state.game_round = 0
    st.session_state.user_score = 0
    st.session_state.computer_score = 0
    st.session_state.game_history = []
    st.rerun()

if play_clicked:
    if current_move not in CLASSES:
        st.warning("Show a hand clearly before playing the round.")
    else:
        computer_move = random.choice(CLASSES)
        result = get_round_winner(current_move, computer_move)
        st.session_state.game_round += 1
        if result == "You win":
            st.session_state.user_score += 1
        elif result == "Computer wins":
            st.session_state.computer_score += 1
        st.session_state.game_history.append(
            (st.session_state.game_round, current_move, computer_move, result)
        )
        st.rerun()

if st.session_state.game_history:
    round_number, user_move, computer_move, result = st.session_state.game_history[-1]
    st.info(
        f"Round {round_number}: You played **{user_move.capitalize()}**, "
        f"computer played **{computer_move.capitalize()}** — **{result}**"
    )

if game_over:
    if st.session_state.user_score > st.session_state.computer_score:
        final_result = "You won the game."
    elif st.session_state.computer_score > st.session_state.user_score:
        final_result = "Computer won the game."
    else:
        final_result = "The game is a draw."
    st.success(final_result)
