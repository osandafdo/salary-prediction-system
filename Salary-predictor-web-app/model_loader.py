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

        self.network = nn.Sequential(

            #First Layer
            nn.Linear(input_size, 128),     # Input layer to 128 neurons
            nn.BatchNorm1d(128),
            nn.ReLU(),       # ReLU activation function
            nn.Dropout(p=0.2),

            # Second Layer
            nn.Linear(128, 64),    # 128 neurons to 64 neurons
            nn.BatchNorm1d(64),
            nn.ReLU(),       # Another ReLU
            nn.Dropout(p=0.2),

            # Third Layer
            nn.Linear(64, 32),    # 64 neurons to 32 neurons
            nn.BatchNorm1d(32),
            nn.ReLU(),       # Another ReLU
            nn.Dropout(p=0.2),

            # Fourth and the final Layer
            nn.Linear(32, 1),     # Output layer to 1 neurons (classes)
        )

    def forward(self, x):
        return self.network(x)

def load_model():
    # Get current directory
    current_dir = os.path.abspath(".")
    parent_dir = Path(current_dir).parent
    #print(parent_dir)
    print(os.path.join(parent_dir, 'model', 'optimized_model.pt'))

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

def predict_salary(model, explainer, input_features, scaler_X, scaler_y):
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

    ## SHAP calculation for the waterfall plot
    shap_values_input = explainer(input_tensor)

    # Change the input vector to 1D array
    input_vector = input_tensor.numpy()[0]            # 1D array for SHAP
    
    # inverse transform the input data to original scale for SHAP plotting
    # input_original = scaler_y.inverse_transform(input_scaled)

    return predicted_usd[0][0], shap_values_input, explainer, input_vector

def get_shap_values_for_model(model, X_train_scaled):

   # Background data
    background_tensor = torch.tensor(X_train_scaled[:500], dtype=torch.float32)

    # Create SHAP Explainer
    explainer = shap.DeepExplainer(model, background_tensor)

    # Explain test data
    X_trained_scaled_tensor = torch.tensor(X_train_scaled[:500], dtype=torch.float32)
    shap_values = explainer(X_trained_scaled_tensor)   # returns Explanation object

    # shap_values is not a tensor, but an Explanation
    print(type(shap_values))  
    print(shap_values.shape)   # works like NumPy shape
    
    return shap_values, explainer