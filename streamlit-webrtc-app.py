import streamlit as st
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import json
import os
from PIL import Image
import logging
from pathlib import Path
import time
from collections import deque
import threading
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoTransformerBase
import av
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="🧘‍♀️ Real-time Yoga Pose Classifier",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced WebRTC Configuration with multiple STUN servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]},
    ]
})

class YogaPoseDetector:
    def __init__(self, model_path='yoga_pose_model_final.pth', label_map_path='label_map.json'):
        """Initialize the Yoga Pose Detector for real-time processing"""
        self.model = None
        self.label_map = None
        self.inv_label_map = None
        self.pose = None
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        
        # Thread-safe predictions
        self.current_prediction = "No Pose"
        self.current_confidence = 0.0
        self.current_top_predictions = []
        self.current_fps = 0.0
        self.pose_detected = False
        self.prediction_lock = threading.Lock()
        
        # Load components
        self._load_label_map(label_map_path)
        self._load_model(model_path)
        self._initialize_mediapipe()
        
    def _load_label_map(self, label_map_path):
        """Load label mapping from JSON file"""
        try:
            if not os.path.exists(label_map_path):
                # Create a default label map if file doesn't exist
                self.label_map = {
                  
                }
                logger.warning(f"Label map file not found: {label_map_path}. Using default labels.")
            else:
                with open(label_map_path, 'r') as f:
                    self.label_map = json.load(f)
                    
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
            logger.info(f"Loaded {len(self.label_map)} pose classes")
            
        except Exception as e:
            logger.error(f"Error loading label map: {e}")
            # Fallback to default
            self.label_map = {
           
            }
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
    
    def _load_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}. Creating dummy model.")
                # Create a dummy model for demo purposes
                self.model = self._create_model_architecture(num_classes=len(self.label_map))
                self.device = torch.device('cpu')
                return
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load the saved checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different save formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    input_size = checkpoint.get('input_size', 132)
                    num_classes = checkpoint.get('num_classes', len(self.label_map))
                    hidden_size = 128
                    
                    self.model = self._create_model_architecture(
                        input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_classes=num_classes
                    )
                    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                elif 'state_dict' in checkpoint:
                    self.model = self._create_model_architecture(num_classes=len(self.label_map))
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model = self._create_model_architecture(num_classes=len(self.label_map))
                    self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint
            
            self.model.eval()
            self.device = device
            
            if device.type == 'cuda':
                self.model = self.model.to(device)
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create dummy model
            self.model = self._create_model_architecture(num_classes=len(self.label_map))
            self.device = torch.device('cpu')
    
    def _create_model_architecture(self, input_size=132, hidden_size=128, num_classes=None):
        """Recreate the model architecture"""
        import torch.nn as nn
        
        class YogaPoseClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(YogaPoseClassifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.dropout = nn.Dropout(0.3)
                self.fc3 = nn.Linear(hidden_size // 2, num_classes)
                
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc3(out)
                return out
        
        if num_classes is None:
            num_classes = len(self.label_map) if self.label_map else 10
        
        return YogaPoseClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe Pose for real-time processing"""
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        keypoints = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            keypoints = [0.0] * (33 * 4)
        return keypoints
    
    def preprocess_keypoints(self, keypoints):
        """Preprocess keypoints for model input"""
        keypoints_array = np.array(keypoints).reshape(-1, 4)
        
        # Filter low visibility keypoints
        low_visibility_mask = keypoints_array[:, 3] < 0.3
        keypoints_array[low_visibility_mask, :3] = 0
        
        keypoints_flat = keypoints_array.flatten()
        keypoints_tensor = torch.tensor([keypoints_flat], dtype=torch.float32)
        
        if hasattr(self, 'device') and self.device.type == 'cuda':
            keypoints_tensor = keypoints_tensor.to(self.device)
            
        return keypoints_tensor
    
    def predict_pose(self, keypoints):
        """Predict yoga pose from keypoints"""
        try:
            keypoints_tensor = self.preprocess_keypoints(keypoints)
            
            with torch.no_grad():
                output = self.model(keypoints_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, pred = torch.max(probabilities, dim=1)
                
                pred_idx = pred.item()
                confidence_score = confidence.item()
                
                # Get top 3 predictions
                top_probs, top_indices = torch.topk(probabilities, min(3, len(self.inv_label_map)), dim=1)
                
                top_predictions = []
                for i in range(min(3, len(self.inv_label_map))):
                    idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    label = self.inv_label_map.get(idx, "Unknown")
                    top_predictions.append((label, prob))
                
                label = self.inv_label_map.get(pred_idx, "Unknown")
                
                # Add to prediction history for smoothing
                self.prediction_history.append((label, confidence_score))
                
                return label, confidence_score, top_predictions
                
        except Exception as e:
            logger.error(f"Error in pose prediction: {e}")
            return "Error", 0.0, []
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction based on history"""
        if not self.prediction_history:
            return "No Pose", 0.0
        
        # Get most recent predictions
        recent_predictions = list(self.prediction_history)[-5:]
        
        # Count occurrences
        pose_counts = {}
        
        for pose, conf in recent_predictions:
            if pose in pose_counts:
                pose_counts[pose]['count'] += 1
                pose_counts[pose]['total_conf'] += conf
            else:
                pose_counts[pose] = {'count': 1, 'total_conf': conf}
        
        # Find most frequent pose
        most_frequent_pose = max(pose_counts.items(), key=lambda x: x[1]['count'])
        pose_name = most_frequent_pose[0]
        avg_confidence = most_frequent_pose[1]['total_conf'] / most_frequent_pose[1]['count']
        
        return pose_name, avg_confidence
    
    def draw_pose_landmarks(self, image, results):
        """Draw pose landmarks on image"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        return image
    
    def update_predictions(self, pose_label, confidence, top_predictions, fps, pose_detected):
        """Thread-safe update of current predictions"""
        with self.prediction_lock:
            self.current_prediction = pose_label
            self.current_confidence = confidence
            self.current_top_predictions = top_predictions
            self.current_fps = fps
            self.pose_detected = pose_detected
    
    def get_current_predictions(self):
        """Thread-safe get current predictions"""
        with self.prediction_lock:
            return (
                self.current_prediction,
                self.current_confidence,
                self.current_top_predictions.copy(),
                self.current_fps,
                self.pose_detected
            )

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = None
        self.show_landmarks = True
        self.confidence_threshold = 0.5
        self.enable_smoothing = True
        self.frame_count = 0
        self.frame_skip = 1
        self.last_frame_time = time.time()
        
    def set_detector(self, detector):
        self.detector = detector
    
    def set_options(self, show_landmarks, confidence_threshold, enable_smoothing, frame_skip):
        self.show_landmarks = show_landmarks
        self.confidence_threshold = confidence_threshold
        self.enable_smoothing = enable_smoothing
        self.frame_skip = frame_skip
    
    def recv(self, frame):
        """Override recv method for better frame handling"""
        if self.detector is None:
            return frame
        
        self.frame_count += 1
        
        # Skip frames for performance
        if self.frame_count % (self.frame_skip + 1) != 0:
            return frame
        
        start_time = time.time()
        
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Ensure the image is not empty
        if img is None or img.size == 0:
            return frame
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.detector.pose.process(rgb_frame)
        
        # Draw landmarks if enabled
        if self.show_landmarks:
            self.detector.draw_pose_landmarks(img, results)
        
        # Extract keypoints and predict if pose detected
        pose_label = "No Pose Detected"
        confidence = 0.0
        top_predictions = []
        pose_detected = False
        
        if results.pose_landmarks:
            pose_detected = True
            keypoints = self.detector.extract_keypoints(results)
            pose_label, confidence, top_predictions = self.detector.predict_pose(keypoints)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                pose_label = "Low Confidence"
            
            # Get smoothed prediction if enabled
            if self.enable_smoothing:
                smoothed_pose, smoothed_conf = self.detector.get_smoothed_prediction()
                display_pose = smoothed_pose
                display_confidence = smoothed_conf
            else:
                display_pose = pose_label
                display_confidence = confidence
        else:
            display_pose = pose_label
            display_confidence = confidence
        
        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        self.detector.fps_counter.append(fps)
        avg_fps = np.mean(self.detector.fps_counter) if self.detector.fps_counter else 0
        
        # Update predictions for UI
        self.detector.update_predictions(display_pose, display_confidence, top_predictions, avg_fps, pose_detected)
        
        # Add text overlay with better positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Add background rectangles for better text visibility
        text_lines = [
            f"Pose: {display_pose}",
            f"Confidence: {display_confidence:.1%}",
            f"FPS: {avg_fps:.1f}"
        ]
        
        for i, text in enumerate(text_lines):
            y_pos = 30 + i * 30
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background rectangle
            cv2.rectangle(img, (5, y_pos - 20), (text_size[0] + 15, y_pos + 5), (0, 0, 0), -1)
            
            # Draw text
            color = (0, 255, 0) if i < 2 else (255, 0, 0)
            cv2.putText(img, text, (10, y_pos), font, font_scale, color, thickness)
        
        # Convert back to av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

@st.cache_resource
def load_detector():
    """Load the pose detector (cached)"""
    return YogaPoseDetector()

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .video-container {
        border: 2px solid #1f77b4;
        border-radius: 15px;
        overflow: hidden;
        background: #000;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown('<h1 class="main-header">🧘‍♀️ Real-time Yoga Pose Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced AI-powered yoga pose detection using your webcam</p>', unsafe_allow_html=True)
    
    # Check browser compatibility
    st.info("📱 **Browser Compatibility**: This app works best in Chrome, Edge, or Firefox. Safari may have limitations.")
    
    # Sidebar configuration
    st.sidebar.markdown("## 🎛️ Camera Settings")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎯 Detection Settings")
    
    # Detection parameters
    confidence_threshold = st.sidebar.slider(
        "🎯 Confidence Threshold", 
        0.0, 1.0, 0.5, 0.05,
        help="Minimum confidence required for pose detection"
    )
    
    show_landmarks = st.sidebar.toggle(
        "🔗 Show Pose Landmarks", 
        value=True,
        help="Display skeleton overlay on video"
    )
    
    enable_smoothing = st.sidebar.toggle(
        "📊 Enable Prediction Smoothing", 
        value=True,
        help="Smooth predictions over time for stability"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ⚡ Performance")
    
    frame_skip = st.sidebar.slider(
        "⏭️ Frame Skip", 
        0, 5, 1,
        help="Skip frames to improve performance (0 = process all frames)"
    )
    
    # Quality preset buttons
    st.sidebar.markdown("### 🚀 Quick Presets")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🏃‍♂️ Performance", help="Optimized for speed"):
            confidence_threshold = 0.3
            frame_skip = 3
            show_landmarks = False
            st.rerun()
    
    with col2:
        if st.button("🎯 Accuracy", help="Optimized for precision"):
            confidence_threshold = 0.7
            frame_skip = 0
            show_landmarks = True
            st.rerun()
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📖 Instructions")
    st.sidebar.markdown("""
    1. 🎥 **Allow Camera Access**: Click allow when prompted
    2. 🧘‍♀️ **Position Yourself**: Stand 6-8 feet from camera
    3. 🤸‍♀️ **Perform Poses**: Try different yoga poses
    4. 📊 **View Results**: Check real-time predictions
    5. ⚙️ **Adjust Settings**: Fine-tune for your needs
    """)
    
    # Main content layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Camera Feed")
        
        # Load detector
        detector = load_detector()
        
        # Video container with custom styling
        video_container = st.container()
        
        with video_container:
            # Enhanced WebRTC streamer with better configuration
            webrtc_ctx = webrtc_streamer(
                key="yoga-pose-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=VideoTransformer,
                media_stream_constraints={
                    "video": {
                        "width": {"min": 480, "ideal": 640, "max": 1280},
                        "height": {"min": 360, "ideal": 480, "max": 720},
                        "frameRate": {"ideal": 15, "max": 30}
                    },
                    "audio": False
                },
                async_processing=False,  # Changed to False for better stability
                video_html_attrs={
                    "style": {
                        "width": "100%", 
                        "height": "auto",
                        "border-radius": "10px",
                        "box-shadow": "0 4px 8px rgba(0,0,0,0.1)"
                    },
                    "controls": False,
                    "autoplay": True,
                    "muted": True,  # Added muted attribute
                }
            )
        
        # Configure video transformer
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_detector(detector)
            webrtc_ctx.video_processor.set_options(
                show_landmarks, confidence_threshold, enable_smoothing, frame_skip
            )
        
        # Connection status
        if webrtc_ctx.state.playing:
            st.success("🟢 Camera is active and streaming")
        else:
            st.warning("🟡 Camera not active - Click 'START' to begin")
        
        # Troubleshooting section
        if not webrtc_ctx.state.playing:
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Camera not working?**
                
                1. **Allow camera permissions** when prompted by your browser
                2. **Refresh the page** and try again
                3. **Check if other apps** are using your camera
                4. **Try a different browser** (Chrome/Edge recommended)
                5. **Ensure HTTPS connection** (required for camera access)
                
                **Still having issues?**
                - Check browser console for error messages
                - Try incognito/private mode
                - Disable browser extensions temporarily
                """)
    
    with col2:
        st.markdown("### 📊 Live Results")
        
        # Results placeholders
        pose_placeholder = st.empty()
        confidence_placeholder = st.empty()
        fps_placeholder = st.empty()
        top_predictions_placeholder = st.empty()
        
        # Initialize session state for results
        if 'last_update_time' not in st.session_state:
            st.session_state.last_update_time = time.time()
        
        # Update results if video is playing
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor and detector:
            try:
                pose, confidence, top_preds, fps, detected = detector.get_current_predictions()
                
                # Current pose display
                with pose_placeholder.container():
                    if detected and pose != "No Pose Detected":
                        confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>🧘‍♀️ Current Pose</h3>
                            <h2 class="{confidence_class}">{pose}</h2>
                            <p>Confidence: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("🔍 No pose detected - position yourself in view of the camera")
                
                # Confidence meter
                with confidence_placeholder.container():
                    if detected:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("🎯 Confidence Score", f"{confidence:.1%}")
                        st.progress(confidence)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # FPS display
                with fps_placeholder.container():
                    color = "🟢" if fps > 15 else "🟡" if fps > 8 else "🔴"
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric(f"{color} Performance", f"{fps:.1f} FPS")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Top predictions
                with top_predictions_placeholder.container():
                    if top_preds and detected:
                        st.markdown("### 🏆 Top 3 Predictions")
                        for i, (label, prob) in enumerate(top_preds[:3]):
                            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            confidence_bar = "▓" * int(prob * 20) + "░" * (20 - int(prob * 20))
                            st.markdown(f"{emoji} **{label}**")
                            st.markdown(f"`{confidence_bar}` {prob:.1%}")
                            
            except Exception as e:
                logger.error(f"Error updating results: {e}")
                st.error("Error updating results. Please refresh the page.")
        else:
            st.info("🎥 Start the camera to see live results here")
    
    # Performance and tips section
    st.markdown("---")
    st.markdown("## 💡 Tips & Troubleshooting")
    
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    
    with tip_col1:
        st.markdown("""
        ### 🚀 **Better Performance**
        - Use **Performance preset** for speed
        - Ensure good **lighting conditions**
        - Close other **camera applications**
        - Use a **modern browser** (Chrome/Edge recommended)
        """)
    
    with tip_col2:
        st.markdown("""
        ### 🎯 **Better Accuracy**
        - Use **Accuracy preset** for precision
        - Stand **6-8 feet** from camera
        - Ensure **full body** is visible
        - Hold poses for **2-3 seconds**
        """)
    
    with tip_col3:
        st.markdown("""
        ### 🔧 **Troubleshooting**
        - **Allow camera permissions** in browser
        - **Refresh page** if camera freezes
        - **Use HTTPS** (required for camera)
        - **Try different browser** if issues persist
        """)
    
    # System information
    with st.expander("ℹ️ System Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Detector Status:** {'✅ Ready' if detector else '❌ Not loaded'}  
            **Available Poses:** {len(detector.label_map) if detector and detector.label_map else 'Unknown'}  
            **Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}
            """)
        with col2:
            st.markdown(f"""
            **WebRTC Status:** {'🟢 Active' if webrtc_ctx.state.playing else '🔴 Inactive'}  
            **Connection State:** {webrtc_ctx.state.ice_connection_state if hasattr(webrtc_ctx.state, 'ice_connection_state') else 'Unknown'}  
            **Model Loaded:** {'✅ Yes' if detector and detector.model else '❌ No'}
            """)

if __name__ == "__main__":
    main()
