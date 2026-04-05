import streamlit as st
import numpy as np
import cv2
import av
import mediapipe as mp

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tflite_runtime.interpreter as tflite

# ---------------------------
# LOAD TFLITE MODEL
# ---------------------------
interpreter = tflite.Interpreter(model_path="sign_language_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

actions = np.load("actions.npy")

SEQUENCE_LENGTH = 30

# ---------------------------
# LOAD MEDIAPIPE
# ---------------------------
pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"),
        running_mode=vision.RunningMode.IMAGE
    )
)

hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE
    )
)

# ---------------------------
# KEYPOINT EXTRACTION
# ---------------------------
def extract_keypoints(pose_result, hand_result):
    pose = np.zeros(33 * 3)
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if pose_result and pose_result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]).flatten()

    if hand_result and hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()

            if hand_result.handedness[i][0].category_name == "Left":
                left_hand = coords
            else:
                right_hand = coords

    return np.concatenate([pose, left_hand, right_hand])


def extract_keypoints_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_result = pose_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)

    return extract_keypoints(pose_result, hand_result)

# ---------------------------
# TFLITE PREDICTION FUNCTION
# ---------------------------
def tflite_predict(sequence):
    input_data = np.expand_dims(sequence, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

# ---------------------------
# VIDEO PROCESSOR
# ---------------------------
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.sequence = []
        self.label = "Starting..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        keypoints = extract_keypoints_from_frame(img)

        self.sequence.append(keypoints)
        self.sequence = self.sequence[-SEQUENCE_LENGTH:]

        if len(self.sequence) == SEQUENCE_LENGTH:
            res = tflite_predict(self.sequence)

            pred_class = np.argmax(res)
            confidence = np.max(res)

            if confidence > 0.6:
                self.label = f"{actions[pred_class]} ({confidence:.2f})"
            else:
                self.label = "Uncertain"

        cv2.putText(img, self.label,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🤟 Real-Time Sign Language Interpreter (TFLite)")

webrtc_streamer(
    key="sign-language",
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
)