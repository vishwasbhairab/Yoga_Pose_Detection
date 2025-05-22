import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Custom dataset class for yoga poses
class YogaPoseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        self.labels = df['label'].values  # Column name is 'label'
        
        # Assuming all columns except 'label' are keypoints
        self.keypoints = df.drop('label', axis=1).values
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        keypoints = self.keypoints[idx].astype(np.float32)  # Convert to float32 for PyTorch
        label = self.labels[idx]
        
        if self.transform:
            keypoints = self.transform(keypoints)
            
        return keypoints, label

# Simple MLP model for yoga pose classification
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

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    device = next(model.parameters()).device  # Get the device from the model
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # No need to track gradients
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')
        
        # Save the best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_yoga_model.pth')
    
    return model

# Main execution code
def main():
    # Load and preprocess data
    train_data = pd.read_csv('train_pose_data.csv')
    test_data = pd.read_csv('test_pose_data.csv')
    
    # Determine the input size (number of keypoints * dimensions per keypoint)
    input_size = train_data.shape[1] - 1  # Subtract 1 for the label column
    
    # Get the number of unique classes
    num_classes = len(train_data['label'].unique())
    print(f"Number of yoga pose classes: {num_classes}")
    
    # Create datasets and dataloaders
    train_dataset = YogaPoseDataset('train_pose_data.csv')
    test_dataset = YogaPoseDataset('test_pose_data.csv')
    
    # Create data loaders with batch processing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the model
    model = YogaPoseClassifier(input_size=input_size, 
                              hidden_size=128, 
                              num_classes=num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30)
    
    # Evaluate the model on test data
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes,
        'input_size': input_size
    }, 'yoga_pose_model_final.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
