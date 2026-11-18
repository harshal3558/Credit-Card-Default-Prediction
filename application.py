from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np

from src.CCDP.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        req_data = request.form if request.form else (request.get_json() or {})
        
        form_data = {
            'LIMIT_BAL': int(req_data.get('LIMIT_BAL', 50000)),
            'SEX': int(req_data.get('SEX', 1)),
            'EDUCATION': int(req_data.get('EDUCATION', 2)),
            'MARRIAGE': int(req_data.get('MARRIAGE', 2)),
            'AGE': int(req_data.get('AGE', 30)),
            'PAY_0': int(req_data.get('PAY_0', 0)),
            'PAY_2': int(req_data.get('PAY_2', 0)),
            'PAY_3': int(req_data.get('PAY_3', 0)),
            'PAY_4': int(req_data.get('PAY_4', 0)),
            'PAY_5': int(req_data.get('PAY_5', 0)),
            'PAY_6': int(req_data.get('PAY_6', 0)),
            'BILL_AMT1': int(req_data.get('BILL_AMT1', 20000)),
            'BILL_AMT2': int(req_data.get('BILL_AMT2', 19000)),
            'BILL_AMT3': int(req_data.get('BILL_AMT3', 18000)),
            'BILL_AMT4': int(req_data.get('BILL_AMT4', 17000)),
            'BILL_AMT5': int(req_data.get('BILL_AMT5', 16000)),
            'BILL_AMT6': int(req_data.get('BILL_AMT6', 15000)),
            'PAY_AMT1': int(req_data.get('PAY_AMT1', 3000)),
            'PAY_AMT2': int(req_data.get('PAY_AMT2', 3000)),
            'PAY_AMT3': int(req_data.get('PAY_AMT3', 3000)),
            'PAY_AMT4': int(req_data.get('PAY_AMT4', 3000)),
            'PAY_AMT5': int(req_data.get('PAY_AMT5', 3000)),
            'PAY_AMT6': int(req_data.get('PAY_AMT6', 3000))
        }
        data = CustomData(**form_data)
        
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        explanation = predict_pipeline.predict_with_explanation(pred_df)
        
        # If client requested JSON (e.g. from fetch/AJAX)
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Accept') == 'application/json':
            return jsonify({
                'status': 'success',
                'results': explanation['prediction_text'],
                'explanation': explanation,
                'form_data': form_data
            })

        return render_template('home.html',
                               results=explanation['prediction_text'],
                               explanation=explanation,
                               form_data=form_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
