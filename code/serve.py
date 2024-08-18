from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join('/opt/ml/model', 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)


@app.route('/ping', methods=['GET'])
def ping():
    return "OK"


@app.route('/invocations', methods=['POST'])
def predict():
    # Read CSV data from request
    csv_data = request.data.decode('utf-8')
    data = pd.read_csv(pd.compat.StringIO(csv_data))

    # Prepare data for Prophet
    data.rename(columns={'date': 'ds', 'value': 'y'}, inplace=True)

    # Forecast
    forecast = model.predict(data[['ds']])
    forecast_json = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_json(orient='records')
    return jsonify(forecast_json)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
