import streamlit as st
import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import json
import os
from PIL import Image
import tempfile
import logging
from pathlib import Path
import time
from collections import deque
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Real-time Yoga Pose Classifier",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class YogaPoseDetector:
    def __init__(self, model_path='yoga_pose_model_final.pth', label_map_path='label_map.json'):
        """
        Initialize the Yoga Pose Detector for real-time processing
        """
        self.model = None
        self.label_map = None
        self.inv_label_map = None
        self.pose = None
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.prediction_history = deque(maxlen=10)
        
        # Load components
        self._load_label_map(label_map_path)
        self._load_model(model_path)
        self._initialize_mediapipe()
        
    def _load_label_map(self, label_map_path):
        """Load label mapping from JSON file"""
        try:
            if not os.path.exists(label_map_path):
                st.error(f"Label map file not found: {label_map_path}")
                st.stop()
                
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
            logger.info(f"Loaded {len(self.label_map)} pose classes")
            
        except Exception as e:
            st.error(f"Error loading label map: {e}")
            st.stop()
    
    def _load_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                st.stop()
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load the saved checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different save formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    input_size = checkpoint.get('input_size', 132)
                    num_classes = checkpoint.get('num_classes', 82)
                    hidden_size = 128
                    
                    if self.label_map and len(self.label_map) != num_classes:
                        num_classes = len(self.label_map)
                    
                    self.model = self._create_model_architecture(
                        input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_classes=num_classes
                    )
                    
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                elif 'state_dict' in checkpoint:
                    self.model = self._create_model_architecture(num_classes=82)
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model = self._create_model_architecture(num_classes=82)
                    self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint
            
            self.model.eval()
            self.device = device
            
            if device.type == 'cuda':
                self.model = self.model.to(device)
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    
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
            static_image_mode=False,  # Set to False for video
            model_complexity=1,
            smooth_landmarks=True,    # Enable smoothing for video
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
                top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
                
                top_predictions = []
                for i in range(3):
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
        recent_predictions = list(self.prediction_history)[-5:]  # Last 5 predictions
        
        # Count occurrences
        pose_counts = {}
        total_confidence = 0
        
        for pose, conf in recent_predictions:
            if pose in pose_counts:
                pose_counts[pose]['count'] += 1
                pose_counts[pose]['total_conf'] += conf
            else:
                pose_counts[pose] = {'count': 1, 'total_conf': conf}
            total_confidence += conf
        
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
    
    def process_frame(self, frame):
        """Process a single frame for pose detection"""
        start_time = time.time()
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Draw landmarks
        annotated_frame = frame.copy()
        self.draw_pose_landmarks(annotated_frame, results)
        
        # Extract keypoints and predict if pose detected
        pose_label = "No Pose Detected"
        confidence = 0.0
        top_predictions = []
        
        if results.pose_landmarks:
            keypoints = self.extract_keypoints(results)
            pose_label, confidence, top_predictions = self.predict_pose(keypoints)
        
        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        self.fps_counter.append(fps)
        avg_fps = np.mean(self.fps_counter)
        
        return annotated_frame, pose_label, confidence, top_predictions, avg_fps, results.pose_landmarks is not None

@st.cache_resource
def load_detector():
    """Load the pose detector (cached)"""
    return YogaPoseDetector()

def main():
    st.title("🧘‍♀️ Real-time Yoga Pose Classifier")
    st.markdown("Real-time yoga pose detection using your webcam!")
    
    # Sidebar
    st.sidebar.header("Camera Controls")
    
    # Camera selection
    camera_index = st.sidebar.selectbox("Select Camera", options=[0, 1, 2], index=0)
    
    # Resolution settings
    resolution = st.sidebar.selectbox(
        "Resolution", 
        options=["640x480", "1280x720", "1920x1080"],
        index=0
    )
    width, height = map(int, resolution.split('x'))
    
    # Detection settings
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    show_landmarks = st.sidebar.checkbox("Show Pose Landmarks", value=True)
    enable_smoothing = st.sidebar.checkbox("Enable Prediction Smoothing", value=True)
    
    # Performance settings
    st.sidebar.header("Performance")
    frame_skip = st.sidebar.slider("Frame Skip (for performance)", 0, 5, 1)
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Click 'Start Camera' to begin
    2. Position yourself in front of the camera
    3. Perform yoga poses to see real-time classification
    4. Adjust settings for better performance
    """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Live Camera Feed")
        
        # Camera controls
        start_camera = st.button("🎥 Start Camera", type="primary")
        stop_camera = st.button("⏹️ Stop Camera")
        
        # Placeholder for video
        video_placeholder = st.empty()
        
    with col2:
        st.header("📊 Live Results")
        
        # Placeholders for results
        pose_placeholder = st.empty()
        confidence_placeholder = st.empty()
        fps_placeholder = st.empty()
        top_predictions_placeholder = st.empty()
    
    # Camera processing
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    if start_camera:
        st.session_state.camera_active = True
    
    if stop_camera:
        st.session_state.camera_active = False
    
    if st.session_state.camera_active:
        try:
            # Load detector
            detector = load_detector()
            
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                st.error(f"Cannot open camera {camera_index}")
                st.session_state.camera_active = False
                return
            
            frame_count = 0
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % (frame_skip + 1) != 0:
                    continue
                
                # Process frame
                processed_frame, pose_label, confidence, top_predictions, fps, pose_detected = detector.process_frame(frame)
                
                # Apply confidence threshold
                if confidence < confidence_threshold:
                    pose_label = "Low Confidence"
                
                # Get smoothed prediction if enabled
                if enable_smoothing and pose_detected:
                    smoothed_pose, smoothed_conf = detector.get_smoothed_prediction()
                    display_pose = smoothed_pose
                    display_confidence = smoothed_conf
                else:
                    display_pose = pose_label
                    display_confidence = confidence
                
                # Show/hide landmarks
                if not show_landmarks:
                    processed_frame = frame
                
                # Add text overlay
                cv2.putText(processed_frame, f"Pose: {display_pose}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Confidence: {display_confidence:.2%}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Convert BGR to RGB for Streamlit
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Update displays
                video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
                # Update results panel
                with pose_placeholder.container():
                    if pose_detected:
                        confidence_color = "green" if display_confidence > 0.7 else "orange" if display_confidence > 0.4 else "red"
                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                            <h4 style="color: #1f77b4; margin: 0;">Current Pose</h4>
                            <h3 style="color: {confidence_color}; margin: 5px 0;">{display_pose}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No pose detected")
                
                with confidence_placeholder.container():
                    if pose_detected:
                        st.metric("Confidence", f"{display_confidence:.2%}")
                        st.progress(display_confidence)
                    else:
                        st.metric("Confidence", "N/A")
                
                with fps_placeholder.container():
                    st.metric("FPS", f"{fps:.1f}")
                
                with top_predictions_placeholder.container():
                    if top_predictions and pose_detected:
                        st.subheader("Top 3 Predictions")
                        for i, (label, prob) in enumerate(top_predictions[:3]):
                            st.text(f"{i+1}. {label}: {prob:.2%}")
                
                # Small delay to prevent overwhelming the UI
                time.sleep(0.01)
            
            cap.release()
            
        except Exception as e:
            st.error(f"Error during camera processing: {str(e)}")
            st.session_state.camera_active = False
    
    else:
        with col1:
            st.info("👆 Click 'Start Camera' to begin real-time pose detection")
        
        with col2:
            st.info("Results will appear here when camera is active")
    
    # Performance tips
    st.markdown("---")
    st.subheader("💡 Performance Tips")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **For Better Performance:**
        - Use lower resolution (640x480)
        - Increase frame skip
        - Ensure good lighting
        """)
    
    with col2:
        st.markdown("""
        **For Better Accuracy:**
        - Stand 6-8 feet from camera
        - Ensure full body is visible
        - Use good lighting
        """)
    
    with col3:
        st.markdown("""
        **Troubleshooting:**
        - Try different camera index (0, 1, 2)
        - Check camera permissions
        - Restart if camera freezes
        """)

if __name__ == "__main__":
    main()