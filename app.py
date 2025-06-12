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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Yoga Pose Classifier",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class YogaPoseDetector:
    def __init__(self, model_path='yoga_pose_model_final.pth', label_map_path='label_map.json'):
        """
        Initialize the Yoga Pose Detector for Streamlit
        """
        self.model = None
        self.label_map = None
        self.inv_label_map = None
        self.pose = None
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
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
        """Initialize MediaPipe Pose"""
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=False,
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
                
                return label, confidence_score, top_predictions
                
        except Exception as e:
            logger.error(f"Error in pose prediction: {e}")
            return "Error", 0.0, []
    
    def draw_pose_landmarks(self, image, results):
        """Draw pose landmarks on image"""
        if results.pose_landmarks:
            annotated_image = image.copy()
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            return annotated_image
        return image
    
    def process_image(self, image):
        """Process a single image for pose detection"""
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process with MediaPipe
        rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        # Extract keypoints and predict
        keypoints = self.extract_keypoints(results)
        pose_label, confidence, top_predictions = self.predict_pose(keypoints)
        
        # Draw landmarks
        annotated_image = self.draw_pose_landmarks(image_cv, results)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        return pose_label, confidence, top_predictions, annotated_image_rgb, results.pose_landmarks is not None

@st.cache_resource
def load_detector():
    """Load the pose detector (cached)"""
    return YogaPoseDetector()

def main():
    st.title("🧘‍♀️ Yoga Pose Classifier")
    st.markdown("Upload an image to detect and classify yoga poses!")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses MediaPipe for pose detection and a trained neural network "
        "to classify yoga poses. Upload an image and see the results!"
    )
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload an image containing a person in a yoga pose
    2. The app will detect pose landmarks
    3. View the predicted pose and confidence score
    4. See top 3 predictions for alternative possibilities
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing a person in a yoga pose"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🔍 Analyze Pose", type="primary"):
                with st.spinner("Analyzing pose..."):
                    try:
                        # Load detector
                        detector = load_detector()
                        
                        # Process image
                        pose_label, confidence, top_predictions, annotated_image, pose_detected = detector.process_image(image)
                        
                        # Store results in session state
                        st.session_state.results = {
                            'pose_label': pose_label,
                            'confidence': confidence,
                            'top_predictions': top_predictions,
                            'annotated_image': annotated_image,
                            'pose_detected': pose_detected
                        }
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.header("📊 Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            if results['pose_detected']:
                # Display annotated image
                st.image(
                    results['annotated_image'], 
                    caption="Pose Detection Results", 
                    use_column_width=True
                )
                
                # Main prediction
                st.subheader("🎯 Primary Prediction")
                
                # Create confidence meter
                confidence_color = "green" if results['confidence'] > 0.7 else "orange" if results['confidence'] > 0.4 else "red"
                
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="color: #1f77b4; margin: 0;">Pose: {results['pose_label']}</h3>
                    <p style="color: {confidence_color}; font-size: 18px; margin: 5px 0;">
                        Confidence: {results['confidence']:.2%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for confidence
                st.progress(results['confidence'])
                
                # Top 3 predictions
                st.subheader("🏆 Top 3 Predictions")
                
                for i, (label, prob) in enumerate(results['top_predictions']):
                    with st.expander(f"{i+1}. {label} ({prob:.2%})"):
                        st.write(f"Confidence: {prob:.2%}")
                        st.progress(prob)
                
                # Confidence interpretation
                st.subheader("📈 Confidence Interpretation")
                if results['confidence'] > 0.8:
                    st.success("🎯 High confidence - The model is very sure about this prediction!")
                elif results['confidence'] > 0.6:
                    st.warning("⚠️ Medium confidence - The prediction is likely correct but consider alternatives.")
                else:
                    st.error("❌ Low confidence - The model is uncertain. Consider the top alternatives.")
                
            else:
                st.warning("⚠️ No pose detected in the image. Please try with a clearer image of a person in a yoga pose.")
        else:
            st.info("👆 Upload an image and click 'Analyze Pose' to see results here.")
    
    # Additional information
    st.markdown("---")
    st.subheader("ℹ️ About the Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pose Detection", "MediaPipe", "Google's ML Kit")
    
    with col2:
        st.metric("Classification", "Neural Network", "PyTorch")
    
    with col3:
        st.metric("Keypoints", "33 Landmarks", "4D Coordinates")
    
    st.info("""
    **How it works:**
    1. **Pose Detection**: MediaPipe identifies 33 key body landmarks
    2. **Feature Extraction**: 4D coordinates (x, y, z, visibility) for each landmark
    3. **Classification**: Neural network predicts the yoga pose from the landmarks
    """)

if __name__ == "__main__":
    main()