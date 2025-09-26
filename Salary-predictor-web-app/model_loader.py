import torch
import torch.nn as nn
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import shap

# Define the model architecture
class FullyConnectedNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 128)
        self.activation_1 = nn.ReLU()
        self.layer_2 = nn.Linear(128, 64)
        self.activation_2 = nn.ReLU()
        self.layer_3 = nn.Linear(64, 32)
        self.activation_3 = nn.ReLU()
        self.layer_4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.activation_1(self.layer_1(x))
        x = self.activation_2(self.layer_2(x))
        x = self.activation_3(self.layer_3(x))
        x = self.layer_4(x)
        return x

def load_model():
    # Get current directory
    current_dir = os.path.abspath(".")
    parent_dir = Path(current_dir).parent
    print(parent_dir)

    # Load the model
    input_size = 41
    model = FullyConnectedNeuralNetwork(input_size)
    model.load_state_dict(torch.load(os.path.join(parent_dir, 'model', 'optimized_model.pt'), map_location=torch.device('cpu')))
    model.eval()
    
    # Load the scalers
    scaler_X = joblib.load(os.path.join(parent_dir, 'model', 'feature_scaler.joblib'))
    scaler_y = joblib.load(os.path.join(parent_dir, 'model', 'target_scaler.joblib'))

    ##### Load the X datasets
    X_train_scaled = np.load(os.path.join(parent_dir, 'model', 'X_train_scaled.npy'))
    X_val_scaled = np.load(os.path.join(parent_dir, 'model', 'X_val_scaled.npy'))
    X_test_scaled = np.load(os.path.join(parent_dir, 'model', 'X_test_scaled.npy'))

    return model, scaler_X, scaler_y, X_train_scaled, X_val_scaled, X_test_scaled

def predict_salary(model, input_features, scaler_X, scaler_y):
    # Scale the input features
    input_scaled = scaler_X.transform(input_features)
    
    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        normalized_pred = model(input_tensor)
    
    # Convert to numpy for descaling
    predicted_scaled_np = normalized_pred.numpy()
    
    # Inverse transform to get the value in original USD units
    predicted_usd = scaler_y.inverse_transform(predicted_scaled_np.reshape(-1, 1))
    
    return predicted_usd[0][0]

def get_shap_values_for_model(input_features, model, scaler_X):
    
    # Use a subset of your training data as background
    background = scaler_X[:100]  # Use 100 samples for background

    # Convert to tensor
    background_tensor = torch.tensor(background, dtype=torch.float32)

    # Create DeepExplainer
    explainer = shap.DeepExplainer(model, background_tensor)

    # Calculate SHAP values
    X_test_scaled_tensor = torch.tensor(scaler_X, dtype=torch.float32)
    shap_values = explainer(X_test_scaled_tensor)

    print(f"SHAP values shape: {np.shape(shap_values)}")
    
    return shap_values, explainer