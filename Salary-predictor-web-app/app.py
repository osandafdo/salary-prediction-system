
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from model_loader import load_model, predict_salary

app = Flask(__name__)

# Load model and scalers when the app starts
try:
    model, scaler_X, scaler_y = load_model()
    print("Model and scalers loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model, scaler_X, scaler_y = None, None, None

@app.route('/')
def home():
    return render_template('index.html')

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

def process_form_data(form_data):
    # Map form fields to the expected feature order
    # This needs to match the exact order your model expects
    feature_order = [
        'ExperienceYears', 'Certifications', 'PreviousCompanies', 'Age', 
        'CompanySize', 'CommuterSupport', 'HealthInsurance', 'FlexibleHours',
        'Gym', 'Bonus', 'StockOptions', 'Retirement', 'Gender_Female',
        'Gender_Male', 'Gender_Non-binary', 'RemoteOnsite_Hybrid',
        'RemoteOnsite_Onsite', 'RemoteOnsite_Remote', 'Industry_Consulting',
        'Industry_Finance', 'Industry_Healthcare', 'Industry_Retail',
        'Industry_Tech', 'Education_Bachelors', 'Education_Diploma',
        'Education_Masters', 'Education_PhD', 'Location_Australia',
        'Location_Germany', 'Location_India', 'Location_SriLanka',
        'Location_Sweden', 'Location_UK', 'Location_USA',
        'JobTitle_DataEngineer', 'JobTitle_DataScientist',
        'JobTitle_FullstackDeveloper', 'JobTitle_LeadEngineer',
        'JobTitle_SeniorSoftwareEngineer', 'JobTitle_SoftwareArchitect',
        'JobTitle_SoftwareEngineer'
    ]
    
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
    app.run(debug=True, port=5001)