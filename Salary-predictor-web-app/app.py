
from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
import io
import base64
from model_loader import load_model, predict_salary, get_shap_values_for_model

import matplotlib
matplotlib.use('Agg')  # set the non-GUI backend first

import matplotlib.pyplot as plt

import shap
shap.initjs()

app = Flask(__name__)

def get_feature_names():
    return [
        "ExperienceYears", "Certifications", "PreviousCompanies", "Age", 
        "CompanySize", "CommuterSupport", "HealthInsurance", "FlexibleHours",
        "Gym", "Bonus", "StockOptions", "Retirement", "Gender_Female",
        "Gender_Male", "Gender_Non-binary", "RemoteOnsite_Hybrid",
        "RemoteOnsite_Onsite", "RemoteOnsite_Remote", "Industry_Consulting",
        "Industry_Finance", "Industry_Healthcare", "Industry_Retail",
        "Industry_Tech", "Education_Bachelors", "Education_Diploma",
        "Education_Masters", "Education_PhD", "Location_Australia",
        "Location_Germany", "Location_India", "Location_SriLanka",
        "Location_Sweden", "Location_UK", "Location_USA",
        "JobTitle_DataEngineer", "JobTitle_DataScientist",
        "JobTitle_FullstackDeveloper", "JobTitle_LeadEngineer",
        "JobTitle_SeniorSoftwareEngineer", "JobTitle_SoftwareArchitect",
        "JobTitle_SoftwareEngineer"
    ]

# Load model and scalers when the app starts
try:
    model, scaler_X, scaler_y, X_train_scaled, X_val_scaled, X_test_scaled = load_model()
    print("Model and scalers loaded successfully")
    # get SHAP values for the prediction input
    shap_values, explainer = get_shap_values_for_model(get_feature_names, model, X_test_scaled)
    print("SHAP loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler_X, scaler_y = None, None, None

@app.route('/test')
def test():
    return 'Test Endpoint is working!'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/shap_dashboard')
def shap_dashboard():
    return render_template('shap_dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler_X is None or scaler_y is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded properly. Please check the model files.'
        })
    
    try:
        # Get data from the form
        data = request.form.to_dict()
        
        # Convert form data to the format expected by the model
        features = process_form_data(data)
        
        # Make prediction
        prediction = predict_salary(model, features, scaler_X, scaler_y)
        
        return jsonify({
            'success': True,
            'prediction': f"${prediction:,.2f}"
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
    
@app.route('/summary_plot', methods=['GET'])
def summary_plot():
   
    # Create summary plot
    plt.subplots(figsize=(12, 8))
    # plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values.values.squeeze(), X_test_scaled, feature_names=get_feature_names(), show=False)
    # plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

@app.route('/bar_plot', methods=['GET'])
def bar_plot():
        
    # Create bar plot
    plt.subplots(figsize=(12, 8))
    # plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values.values.squeeze(), X_test_scaled, feature_names=get_feature_names(), plot_type="bar", show=False)
    # plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

@app.route('/waterfall_plot', methods=['GET'])
def waterfall_plot():
        
    # Create waterfall plot for the first sample
    exp = shap.Explanation(
        values=shap_values.values[0].squeeze(),
        base_values=explainer.expected_value,
        data=X_test_scaled[0],
        feature_names=get_feature_names()
    )
    
    plt.subplots(figsize=(12, 8))
    # plt.figure(figsize=(12, 8))
    shap.plots.waterfall(exp, show=False)
    # plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

def process_form_data(form_data):
    # Map form fields to the expected feature order
    # This needs to match the exact order your model expects
    feature_order = get_feature_names()
    
    # Create feature array in the correct order
    features = []
    for feature in feature_order:
        if feature in form_data:
            # Handle categorical features (one-hot encoded)
            if form_data[feature] == '1':
                features.append(1.0)
            else:
                features.append(0.0)
        else:
            # For numerical features, use the value directly
            features.append(float(form_data.get(feature, 0)))
    
    return np.array([features])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=False)