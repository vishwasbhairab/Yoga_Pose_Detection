import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import json
import os
import time
from collections import deque

# Page config
st.set_page_config(page_title="Yoga Pose Classifier", page_icon="🧘‍♂️")
st.title("🧘‍♂️ Real-time Yoga Pose Classifier")
st.markdown("Web-based yoga pose detection using your webcam!")

class YogaPoseDetector:
    def __init__(self, model_path='yoga_pose_model_final.pth', label_map_path='label_map.json'):
        self.model = None
        self.label_map = None
        self.inv_label_map = None
        self.pose = None
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.fps_counter = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)

        self._load_label_map(label_map_path)
        self._load_model(model_path)
        self._initialize_mediapipe()

    def _load_label_map(self, label_map_path):
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def _load_model(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)
        self.device = device

        if 'model_state_dict' in checkpoint:
            model = self._create_model_architecture(
                input_size=checkpoint.get('input_size', 132),
                hidden_size=128,
                num_classes=checkpoint.get('num_classes', len(self.label_map))
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = checkpoint

        self.model = model.to(device)
        self.model.eval()

    def _create_model_architecture(self, input_size=132, hidden_size=128, num_classes=82):
        import torch.nn as nn
        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.dropout = nn.Dropout(0.3)
                self.fc3 = nn.Linear(hidden_size // 2, num_classes)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                return self.fc3(x)

        return Classifier()

    def _initialize_mediapipe(self):
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                      smooth_landmarks=True, enable_segmentation=False,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def extract_keypoints(self, results):
        if results.pose_landmarks:
            return [coord for lm in results.pose_landmarks.landmark for coord in (lm.x, lm.y, lm.z, lm.visibility)]
        return [0.0] * (33 * 4)

    def preprocess_keypoints(self, keypoints):
        array = np.array(keypoints).reshape(-1, 4)
        array[array[:, 3] < 0.3, :3] = 0
        flat = array.flatten()
        tensor = torch.tensor([flat], dtype=torch.float32).to(self.device)
        return tensor

    def predict_pose(self, keypoints):
        with torch.no_grad():
            input_tensor = self.preprocess_keypoints(keypoints)
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            label = self.inv_label_map.get(pred.item(), "Unknown")
            return label, conf.item()

    def process(self, frame):
        img = frame.copy()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        keypoints = self.extract_keypoints(results)
        label, conf = self.predict_pose(keypoints)
        cv2.putText(img, f"Pose: {label} ({conf:.2%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

# Load once
@st.cache_resource
def get_detector():
    return YogaPoseDetector()

# WebRTC transformer
class PoseDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = get_detector()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        out = self.detector.process(img)
        return out

# Start streaming
webrtc_streamer(
    key="yoga",
    video_transformer_factory=PoseDetectionTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
