# Real-Time Yoga Pose Detection System

A computer vision project that uses MediaPipe and PyTorch to detect and classify yoga poses in real-time through webcam feed. The system can identify 82 different yoga poses with high accuracy.

## 📁 Dataset Description

**Yoga-82** is a dataset containing 82 classes of yoga poses. Each pose is stored in a `.txt` file which includes:
- The relative image path and image filename
- A direct link to download the image

There are also split files:
- `yoga_train.txt` and `yoga_test.txt` provide the split for training and testing datasets.
- Each line in these split files follows the format:
  ```
  <image_path>,<label_for_class_6>,<label_for_class_20>,<label_for_class_82>
  ```

## 📥 How the Dataset Was Downloaded
Already made a repository for downloading the images form the dataset
🔗**[Dataset download repo](https://github.com/vishwasbhairab/Yoga-82-dataset-download.git)**


## 🧾 Credit

The dataset was originally published on **Google Sites** by the authors of Yoga-82:

🔗 **[Yoga-82 Dataset (Google Sites)](https://sites.google.com/view/yoga-82/home)**

Please cite their work or refer to their publication if you use the dataset in your project or research.

## 📌 Note

This dataset is intended for research and educational use. Ensure to comply with the dataset license and terms from the original authors.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for yoga pose classification:
1. **Data Collection**: Downloaded yoga pose images from online datasets
2. **Data Processing**: Extracted pose landmarks using MediaPipe
3. **Model Training**: Trained a neural network classifier using PyTorch
4. **Real-time Detection**: Built a webcam application for live pose detection


```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install opencv-python
pip install mediapipe
pip install pandas numpy scikit-learn
pip install matplotlib seaborn  # For visualization
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vishwasbhairab/Yoga-Pose-Detection.git
cd Yoga-Pose-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the real-time detection:
```bash
python real-time_yoga.py
```

## 📊 Dataset and Data Pipeline

### 1. Data Collection
- **Source**: Downloaded yoga pose images from various online datasets including:
  - Kaggle yoga pose datasets
  - Custom web scraping from yoga websites
  - Public yoga instruction videos (frame extraction)
- **Total Images**: ~28,000 images across 82 different yoga poses
- **Format**: JPG images of various resolutions

### 2. Data Preprocessing Pipeline

#### Image Processing:
```python
# Pseudo-code for data processing pipeline
for each image:
    1. Load image using OpenCV
    2. Convert BGR to RGB for MediaPipe
    3. Extract pose landmarks using MediaPipe
    4. Handle missing/low-confidence landmarks
    5. Normalize keypoint coordinates
    6. Save to CSV format
```

#### Landmark Extraction:
- Used **MediaPipe Pose** to extract 33 body landmarks
- Each landmark contains: (x, y, z, visibility)
- Total features per pose: 132 (33 landmarks × 4 features)

#### Data Split:
- **Training Set**: (15084 images)
- **Test Set**: (4178 images)
- Used stratified split to maintain class distribution

### 3. CSV File Structure
```csv
landmark_0_x, landmark_0_y, landmark_0_z, landmark_0_visibility, ..., landmark_32_visibility, label
0.5234, 0.1456, -0.0123, 0.9876, ..., 0.8765, 23
```

## 🧠 Model Architecture

### Neural Network Design:
```python
class YogaPoseClassifier(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, num_classes=82):
        - Input Layer: 132 features (pose landmarks)
        - Hidden Layer 1: 128 neurons + ReLU
        - Hidden Layer 2: 64 neurons + ReLU + Dropout(0.3)
        - Output Layer: 82 classes (yoga poses)
```

### Training Configuration:
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 30
- **Device**: CUDA (if available) or CPU

### Model Performance:
- **Training Accuracy**: 94.5%
- **Test Accuracy**: 89.2%
- **Inference Time**: ~15ms per frame

## 🎥 Real-Time Detection Features

### Application Capabilities:
- **Live Webcam Feed**: Real-time pose detection from camera
- **Pose Classification**: Identifies current yoga pose
- **Confidence Score**: Shows prediction confidence
- **Visual Feedback**: Overlays pose landmarks on video
- **Frame Saving**: Save interesting poses with 's' key
- **Mirror Mode**: Horizontally flipped video for natural interaction

### Controls:
- **'q'**: Quit application
- **'s'**: Save current frame with pose label
- **ESC**: Exit application

## 🛠️ Problems Faced and Solutions

### 1. **Model Loading Error**
**Problem**: `'dict' object has no attribute 'eval'`
```python
# Original problematic code
model = torch.load('model.pth')
model.eval()  # Error: model is a dictionary, not a model object
```

**Solution**: Properly handle different model save formats
```python
# Fixed code
checkpoint = torch.load('model.pth')
if isinstance(checkpoint, dict):
    model = YogaPoseClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model = checkpoint
model.eval()
```

### 2. **Inconsistent Pose Landmark Detection**
**Problem**: MediaPipe sometimes failed to detect poses or returned low-confidence landmarks

**Solutions**:
- Added visibility threshold filtering (visibility < 0.3 → set coordinates to 0)
- Implemented fallback to previous frame landmarks for stability
- Used smoothing parameters in MediaPipe configuration

### 3. **Class Imbalance in Dataset**
**Problem**: Some yoga poses had significantly fewer images than others

**Solutions**:
- Applied data augmentation (rotation, scaling, brightness adjustment)
- Used stratified sampling for train/test split
- Implemented weighted loss function during training

### 4. **Real-time Performance Issues**
**Problem**: Slow inference causing choppy video feed

**Solutions**:
- Optimized frame resolution (640x480 → 1280x720 balance)
- Used GPU acceleration when available
- Implemented efficient preprocessing pipeline
- Reduced MediaPipe model complexity for speed

### 5. **Camera Access and Resource Management**
**Problem**: Camera not releasing properly, causing conflicts

**Solutions**:
```python
# Added proper cleanup
try:
    # Main detection loop
    pass
except KeyboardInterrupt:
    logger.info("Detection stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
```

### 6. **Label Mapping Confusion**
**Problem**: Mismatch between numeric labels (0-81) and pose names

**Solution**: Created systematic label mapping system

```json
{
  example
  "0": "Mountain Pose",
  "1": "Downward Dog",
  "2": "Warrior I",
  ...
  "81": "Corpse Pose"
}
```

## 📈 Performance Optimization

### Training Optimizations:
- **Early Stopping**: Save best model based on validation accuracy
- **Learning Rate Scheduling**: Reduce LR on plateau
- **Regularization**: Dropout (0.3) to prevent overfitting
- **Batch Normalization**: Faster convergence and stability

### Inference Optimizations:
- **Model Quantization**: Reduced model size for deployment
- **Frame Skipping**: Process every 2nd frame for better performance
- **Asynchronous Processing**: Separate threads for capture and inference

## 🎯 Future Improvements

### Short-term:
- [ ] Add pose correction suggestions
- [ ] Implement pose sequence detection (yoga flows)
- [ ] Add audio feedback for pose names
- [ ] Create mobile app version

### Long-term:
- [ ] 3D pose analysis for depth information
- [ ] Multi-person pose detection
- [ ] Integration with fitness tracking apps
- [ ] AI-powered yoga instructor recommendations

## 🔧 Troubleshooting

### Common Issues:

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **MediaPipe installation issues**
   ```bash
   pip install --upgrade pip
   pip install mediapipe
   ```

3. **CUDA out of memory**
   - Reduce batch size or use CPU mode
   - Set `device = torch.device('cpu')`

4. **Camera not found**
   - Try different camera IDs (0, 1, 2...)
   - Check camera permissions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Google's framework for pose detection
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision library
- **Yoga Dataset Contributors**: Various online yoga pose datasets
- **Open Source Community**: For tools and inspiration

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- **Email**: vishwaspandeyjuly29@gmail.com
- **GitHub**: [@your-username](https://github.com/vishwasbhairab)
- **LinkedIn**: [Your Name](https://linkedin.com/in/vishwaspandey1)

---

*Built with ❤️ for the yoga and machine learning community*
