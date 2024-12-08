from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model from the pickle file
with open('random_forest_model_xgb.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form data
    features = [float(request.form['channel_sales']),
                float(request.form['cons_gas_12m']),
                float(request.form['forecast_cons_12m']),
                float(request.form['forecast_discount_energy']),
                float(request.form['forecast_meter_rent_12m']),
                float(request.form['nb_prod_act']),
                float(request.form['net_margin']),
                float(request.form['num_years_antig']),
                float(request.form['pow_max'])]  # Adjust based on your features
    
    # Convert to numpy array and reshape for model prediction
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return result to HTML
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug = True)