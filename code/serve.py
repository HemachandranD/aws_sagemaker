from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('/opt/ml/model', 'prophet_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)


@app.route('/ping', methods=['GET'])
def ping():
    return "OK"


@app.route('/invocations', methods=['POST'])
def predict():
    # Load JSON data from request and convert to DataFrame
    input_json = request.get_json()
    data = pd.DataFrame(input_json)

    # Forecast
    forecast = model.predict(data[['ds']])
    forecast_json = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json(orient='records')
    return jsonify(forecast_json)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
