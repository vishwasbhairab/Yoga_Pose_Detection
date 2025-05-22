import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import json
import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YogaPoseDetector:
    def __init__(self, model_path='yoga_pose_model_final.pth', label_map_path='label_map.json'):
        """
        Initialize the Yoga Pose Detector
        
        Args:
            model_path (str): Path to the trained PyTorch model
            label_map_path (str): Path to the label mapping JSON file
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
                raise FileNotFoundError(f"Label map file not found: {label_map_path}")
                
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            self.inv_label_map = {v: k for k, v in self.label_map.items()}
            logger.info(f"Loaded {len(self.label_map)} pose classes")
            
        except Exception as e:
            logger.error(f"Error loading label map: {e}")
            sys.exit(1)
    
    def _load_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
            # Load the saved checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different save formats
            if isinstance(checkpoint, dict):
                # Your model is saved as a checkpoint dictionary
                if 'model_state_dict' in checkpoint:
                    logger.info("Loading model from checkpoint dictionary")
                    
                    # Get model parameters from checkpoint
                    input_size = checkpoint.get('input_size', 132)  # Default to 132 (33 landmarks * 4 features)
                    num_classes = checkpoint.get('num_classes', 82)  # Your model has 82 classes
                    hidden_size = 128  # This is fixed in your training code
                    
                    # Verify num_classes matches label_map if available
                    if self.label_map and len(self.label_map) != num_classes:
                        logger.warning(f"Label map has {len(self.label_map)} classes but model expects {num_classes}")
                        num_classes = len(self.label_map)
                    
                    logger.info(f"Model parameters: input_size={input_size}, num_classes={num_classes}, hidden_size={hidden_size}")
                    
                    # Create model with correct architecture
                    self.model = self._create_model_architecture(
                        input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_classes=num_classes
                    )
                    
                    # Load the state dict
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                elif 'state_dict' in checkpoint:
                    # Alternative checkpoint format
                    logger.info("Loading model from state_dict")
                    self.model = self._create_model_architecture(num_classes=82)
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the dict itself is the state_dict
                    logger.info("Loading model from direct state_dict")
                    self.model = self._create_model_architecture(num_classes=82)
                    self.model.load_state_dict(checkpoint)
            else:
                # Case 2: Saved as complete model object
                logger.info("Loading complete model object")
                self.model = checkpoint
            
            self.model.eval()
            self.device = device
            
            # Move model to device if using GPU
            if device.type == 'cuda':
                self.model = self.model.to(device)
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error("Make sure your model file contains the correct checkpoint format")
            raise e
    
    def _create_model_architecture(self, input_size=132, hidden_size=128, num_classes=None):
        """
        Recreate your exact model architecture from the training code.
        This matches the YogaPoseClassifier from your training script.
        """
        import torch.nn as nn
        
        class YogaPoseClassifier(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(YogaPoseClassifier, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
                self.relu = nn.ReLU()  # Activation function
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Second layer with reduced size
                self.dropout = nn.Dropout(0.3)  # Dropout for regularization
                self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # Output layer
                
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc3(out)
                return out
        
        # Use provided parameters or defaults
        if num_classes is None:
            num_classes = len(self.label_map) if self.label_map else 10
        
        model = YogaPoseClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
        
        logger.info(f"Created YogaPoseClassifier with input_size={input_size}, hidden_size={hidden_size}, num_classes={num_classes}")
        return model
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe Pose with optimized settings"""
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between accuracy and speed
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints(self, results):
        """
        Extract keypoints from MediaPipe results
        
        Args:
            results: MediaPipe pose detection results
            
        Returns:
            list: Flattened keypoints array
        """
        keypoints = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            keypoints = [0.0] * (33 * 4)  # 33 landmarks, 4 values each
        return keypoints
    
    def preprocess_keypoints(self, keypoints):
        """
        Preprocess keypoints (normalize, filter, etc.)
        
        Args:
            keypoints (list): Raw keypoints
            
        Returns:
            torch.Tensor: Preprocessed keypoints tensor
        """
        # Convert to numpy for easier manipulation
        keypoints_array = np.array(keypoints).reshape(-1, 4)  # 33 landmarks x 4 features
        
        # Filter out low-visibility keypoints (optional)
        # You might want to set coordinates to 0 for low-visibility points
        low_visibility_mask = keypoints_array[:, 3] < 0.3  # visibility threshold
        keypoints_array[low_visibility_mask, :3] = 0  # Set x, y, z to 0 for low visibility points
        
        # Normalize coordinates (optional - depends on your training data)
        # keypoints_array[:, :2] = keypoints_array[:, :2] * 2 - 1  # Normalize x, y to [-1, 1]
        
        # Flatten back to 1D
        keypoints_flat = keypoints_array.flatten()
        
        # Convert to tensor
        keypoints_tensor = torch.tensor([keypoints_flat], dtype=torch.float32)
        
        # Move to device if using GPU
        if hasattr(self, 'device') and self.device.type == 'cuda':
            keypoints_tensor = keypoints_tensor.to(self.device)
            
        return keypoints_tensor
    
    def predict_pose(self, keypoints):
        """
        Predict yoga pose from keypoints
        
        Args:
            keypoints (list): Extracted keypoints
            
        Returns:
            tuple: (predicted_label, confidence_score)
        """
        try:
            keypoints_tensor = self.preprocess_keypoints(keypoints)
            
            with torch.no_grad():
                output = self.model(keypoints_tensor)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1)
                confidence, pred = torch.max(probabilities, dim=1)
                
                pred_idx = pred.item()
                confidence_score = confidence.item()
                
                # Get label name
                if pred_idx in self.inv_label_map:
                    label = self.inv_label_map[pred_idx]
                else:
                    label = "Unknown"
                    logger.warning(f"Unknown prediction index: {pred_idx}")
                
                return label, confidence_score
                
        except Exception as e:
            logger.error(f"Error in pose prediction: {e}")
            return "Error", 0.0
    
    def draw_pose_info(self, frame, pose_label, confidence, results):
        """
        Draw pose information on the frame
        
        Args:
            frame: OpenCV frame
            pose_label (str): Predicted pose label
            confidence (float): Prediction confidence
            results: MediaPipe results
        """
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        # Create info box background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw pose information
        cv2.putText(frame, f'Pose: {pose_label}', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save frame", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self, camera_id=0, save_frames=False):
        """
        Run real-time yoga pose detection
        
        Args:
            camera_id (int): Camera device ID
            save_frames (bool): Whether to enable frame saving
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            logger.error(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting webcam. Press 'q' to exit, 's' to save frame.")
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose detection
                results = self.pose.process(rgb_frame)
                
                # Extract keypoints and predict
                keypoints = self.extract_keypoints(results)
                pose_label, confidence = self.predict_pose(keypoints)
                
                # Draw information on frame
                self.draw_pose_info(frame, pose_label, confidence, results)
                
                # Display frame
                cv2.imshow('Yoga Pose Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_frames:
                    # Save current frame
                    filename = f'pose_frame_{frame_count}_{pose_label}_{confidence:.2f}.jpg'
                    cv2.imwrite(filename, frame)
                    logger.info(f"Saved frame: {filename}")
                
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        except Exception as e:
            logger.error(f"Error during detection: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released and windows closed")

def main():
    """Main function to run the yoga pose detector"""
    try:
        # Initialize detector
        detector = YogaPoseDetector()
        
        # Run detection
        detector.run_detection(camera_id=0, save_frames=True)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()