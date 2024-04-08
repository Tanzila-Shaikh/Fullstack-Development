from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)


# Load the pre-trained model
model = joblib.load('model.pkl')

# Dictionary to map categorical values to numerical values
categorical_mapping = {
    'Usually': 3,
    'Sometimes': 2,
    'Seldom': 1,
    'Most often': 0,
    'Yes': 1,
    'No': 0
}

class FeatureExtractor:
    def __init__(self, categorical_mapping):
        self.categorical_mapping = categorical_mapping

    def extract_features(self, form_data):
        features = []
        features.append(self._map_categorical(form_data['sadness']))
        features.append(self._map_categorical(form_data['euphoric']))
        features.append(self._map_categorical(form_data['exhausted']))
        features.append(self._map_categorical(form_data['sleep_disorder']))
        features.append(self._map_categorical(form_data['mood_swing']))
        features.append(self._map_categorical(form_data['suicidal_thoughts']))
        features.append(self._map_categorical(form_data['anorexia']))
        features.append(self._map_categorical(form_data['authority_respect']))
        features.append(self._map_categorical(form_data['try_explanation']))
        features.append(self._map_categorical(form_data['aggressive_response']))
        features.append(self._map_categorical(form_data['ignore_move_on']))
        features.append(self._map_categorical(form_data['nervous_breakdown']))
        features.append(self._map_categorical(form_data['admit_mistakes']))
        features.append(self._map_categorical(form_data['overthinking']))
        features.append(self._map_categorical(form_data['sexual_activity']))
        features.append(self._map_categorical(form_data['concentration']))
        features.append(self._map_categorical(form_data['optimism']))
        return features

    def _map_categorical(self, value):
        return self.categorical_mapping.get(value, 0)  # Default to 0 if value not found in mapping

feature_extractor = FeatureExtractor(categorical_mapping)

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST']) # type: ignore
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            form_data = request.form.to_dict()

            # Extract features
            features = feature_extractor.extract_features(form_data)

            # Prepare input data for prediction
            X = np.array([features])

            # Use the model to make predictions
            prediction = model.predict(X)
            
            # Get prediction probabilities
            prediction_proba = model.predict_proba(X)
            
            # Generate pie chart
            fig, ax = plt.subplots()
            ax.pie(prediction_proba[0]*100, labels=model.classes_, autopct='%1.1f%%')
            ax.set_title(f" The patient has {(prediction)} mental disorder.")
            
            # Save pie chart to a bytes buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()

            # Render the result template with prediction and pie chart
            return render_template('result.html', prediction=prediction[0], pie_chart=img_str)
        except KeyError:
            # Handle error when input value is not found in the mapping
            return "Invalid input value. Please select a valid option."

if __name__ == '__main__':
    app.run(debug=True)
