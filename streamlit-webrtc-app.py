import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import json
import os
import time
from collections import deque
from datetime import datetime
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Yoga Instructor", 
    page_icon="🧘‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .pose-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .stat-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        min-width: 120px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧘‍♂️ AI Yoga Instructor</h1>', unsafe_allow_html=True)

class YogaPoseDetector:
    def __init__(self, model_path='yoga_pose_model_final.pth', label_map_path='label_map.json'):
        self.model = None
        self.label_map = None
        self.inv_label_map = None
        self.pose = None
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        self.pose_stats = {}
        self.session_start = time.time()
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Thread safety
        self.lock = threading.Lock()
        
        self._load_label_map(label_map_path)
        self._initialize_model(model_path)
        self._initialize_mediapipe()

    def _load_label_map(self, label_map_path):
        """Load label mapping from file"""
        try:
            if os.path.exists(label_map_path):
                with open(label_map_path, 'r') as f:
                    self.label_map = json.load(f)
                logger.info(f"Loaded {len(self.label_map)} pose labels")
                self.inv_label_map = {v: k for k, v in self.label_map.items()}
            else:
                logger.error(f"Label map not found at {label_map_path}")
                st.error(f"Required file not found: {label_map_path}")
                st.stop()
        except Exception as e:
            logger.error(f"Error loading label map: {e}")
            st.error(f"Failed to load label map: {e}")
            st.stop()

    def _initialize_model(self, model_path):
        """Initialize the pose classification model"""
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if 'model_state_dict' in checkpoint:
                    self.model = self._create_model_architecture(
                        input_size=checkpoint.get('input_size', 132),
                        hidden_size=checkpoint.get('hidden_size', 128),
                        num_classes=checkpoint.get('num_classes', len(self.label_map) if self.label_map else 20)
                    )
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model = checkpoint
                
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Model loaded successfully on {self.device}")
            else:
                logger.error(f"Model not found at {model_path}")
                st.error(f"Required file not found: {model_path}")
                st.stop()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"Failed to load model: {e}")
            st.stop()

    def _create_model_architecture(self, input_size=132, hidden_size=128, num_classes=20):
        """Create the neural network architecture"""
        if self.label_map:
            num_classes = len(self.label_map)
            
        class YogaPoseClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.bn1 = nn.BatchNorm1d(hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.bn2 = nn.BatchNorm1d(hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
                self.dropout = nn.Dropout(0.3)
                self.fc4 = nn.Linear(hidden_size // 4, num_classes)

            def forward(self, x):
                x = F.relu(self.bn1(self.fc1(x)))
                x = F.relu(self.bn2(self.fc2(x)))
                x = F.relu(self.fc3(x))
                x = self.dropout(x)
                return self.fc4(x)

        return YogaPoseClassifier(input_size, hidden_size, num_classes)

    def _initialize_mediapipe(self):
        """Initialize MediaPipe pose detection"""
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, results):
        """Extract pose keypoints from MediaPipe results"""
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            return keypoints
        return [0.0] * (33 * 4)  # 33 landmarks * 4 coordinates

    def preprocess_keypoints(self, keypoints):
        """Preprocess keypoints for model input"""
        array = np.array(keypoints).reshape(-1, 4)
        
        # Filter out low-confidence keypoints
        array[array[:, 3] < 0.3, :3] = 0
        
        # Normalize coordinates
        array[:, :2] = array[:, :2] - array[:, :2].mean(axis=0)
        
        flat = array.flatten()
        tensor = torch.tensor([flat], dtype=torch.float32).to(self.device)
        return tensor

    def predict_pose(self, keypoints):
        """Predict yoga pose from keypoints"""
        try:
            with torch.no_grad():
                input_tensor = self.preprocess_keypoints(keypoints)
                output = self.model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                
                pose_name = self.inv_label_map.get(pred.item(), "Unknown")
                confidence = conf.item()
                
                # Update prediction history
                self.prediction_history.append((pose_name, confidence))
                
                # Get stabilized prediction
                return self._get_stable_prediction()
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0

    def _get_stable_prediction(self):
        """Get stabilized prediction from history"""
        if not self.prediction_history:
            return "Unknown", 0.0
        
        # Use majority voting for stability
        pose_votes = {}
        total_conf = 0
        
        for pose, conf in self.prediction_history:
            if pose not in pose_votes:
                pose_votes[pose] = []
            pose_votes[pose].append(conf)
            total_conf += conf
        
        # Find most frequent pose with high confidence
        best_pose = max(pose_votes.keys(), 
                       key=lambda x: len(pose_votes[x]) * np.mean(pose_votes[x]))
        best_conf = np.mean(pose_votes[best_pose])
        
        return best_pose, best_conf

    def update_stats(self, pose_name):
        """Update pose statistics"""
        with self.lock:
            if pose_name not in self.pose_stats:
                self.pose_stats[pose_name] = 0
            self.pose_stats[pose_name] += 1

    def get_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
            return fps
        return 0

    def process_frame(self, frame):
        """Process a single frame"""
        try:
            self.frame_count += 1
            img = frame.copy()
            
            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Extract and predict pose
                keypoints = self.extract_keypoints(results)
                pose_name, confidence = self.predict_pose(keypoints)
                
                # Update statistics
                if confidence > 0.5:  # Only count high-confidence predictions
                    self.update_stats(pose_name)
                
                # Draw pose information
                self._draw_pose_info(img, pose_name, confidence)
            else:
                cv2.putText(img, "No pose detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw FPS
            fps = self.get_fps()
            cv2.putText(img, f"FPS: {fps:.1f}", (img.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return img
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

    def _draw_pose_info(self, img, pose_name, confidence):
        """Draw pose information on the frame"""
        # Format pose name
        display_name = pose_name.replace('_', ' ').title()
        
        # Choose color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange
        
        # Draw background rectangle
        text = f"{display_name}: {confidence:.1%}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(img, (10, 10), (text_width + 20, text_height + 20), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Streamlit WebRTC transformer
class YogaPoseTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = YogaPoseDetector()
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = self.detector.process_frame(img)
        return processed_img

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # WebRTC configuration
    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # Settings
    st.subheader("⚙️ Settings")
    show_landmarks = st.checkbox("Show pose landmarks", value=True)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.1)
    
    # Session info
    st.subheader("📊 Session Info")
    if 'detector' in st.session_state:
        detector = st.session_state['detector']
        session_time = time.time() - detector.session_start
        st.metric("Session duration", f"{session_time/60:.1f} min")
        st.metric("Frames processed", detector.frame_count)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Pose Detection")
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="yoga-pose-detection",
        video_transformer_factory=YogaPoseTransformer,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        },
        async_processing=True,
    )

with col2:
    st.subheader("📈 Statistics")
    
    # Placeholder for real-time stats
    stats_placeholder = st.empty()
    
    # Instructions
    st.subheader("📝 Instructions")
    st.markdown("""
    1. **Allow camera access** when prompted
    2. **Position yourself** in front of the camera
    3. **Hold yoga poses** for best detection
    4. **Check your form** using the pose landmarks
    """)

# Tips section
st.subheader("💡 Tips for Better Detection")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🏠 Environment**
    - Good lighting
    - Clear background
    - Stable camera
    """)

with col2:
    st.markdown("""
    **👤 Positioning**
    - Full body visible
    - Face the camera
    - Hold poses steady
    """)

with col3:
    st.markdown("""
    **🧘‍♀️ Practice**
    - Start with basic poses
    - Focus on alignment
    - Breathe deeply
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Made with ❤️ using Streamlit, MediaPipe, and PyTorch<br>
    <small>Real-time yoga pose detection for better practice</small>
</div>
""", unsafe_allow_html=True)

# Performance monitoring (runs in background)
if webrtc_ctx.state.playing:
    st.session_state['detector'] = webrtc_ctx.video_transformer.detector
