import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

# Test Prediction Pipeline
try:
    from src.CCDP.pipelines.prediction_pipeline import CustomData, PredictPipeline
    custom_data = CustomData(
        LIMIT_BAL=20000, SEX=2, EDUCATION=2, MARRIAGE=1, AGE=24,
        PAY_0=2, PAY_2=2, PAY_3=-1, PAY_4=-1, PAY_5=-2, PAY_6=-2,
        BILL_AMT1=3913, BILL_AMT2=3102, BILL_AMT3=689, BILL_AMT4=0, BILL_AMT5=0, BILL_AMT6=0,
        PAY_AMT1=0, PAY_AMT2=689, PAY_AMT3=0, PAY_AMT4=0, PAY_AMT5=0, PAY_AMT6=0
    )
    df = custom_data.get_data_as_data_frame()
    pipeline = PredictPipeline()
    pred = pipeline.predict(df)
    print(f"[TEST 1 SUCCESS] Prediction Output: {pred}")
except Exception as e:
    print(f"[TEST 1 FAILED] Prediction Error: {e}")
    sys.exit(1)

# Test Application Flask Endpoint
try:
    from application import app
    client = app.test_client()
    res_index = client.get("/")
    assert res_index.status_code == 200, f"Index returned {res_index.status_code}"
    
    res_get_predict = client.get("/predictdata")
    assert res_get_predict.status_code == 200, f"GET predictdata returned {res_get_predict.status_code}"
    
    post_payload = {
        'LIMIT_BAL': '20000', 'SEX': '2', 'EDUCATION': '2', 'MARRIAGE': '1', 'AGE': '24',
        'PAY_0': '2', 'PAY_2': '2', 'PAY_3': '-1', 'PAY_4': '-1', 'PAY_5': '-2', 'PAY_6': '-2',
        'BILL_AMT1': '3913', 'BILL_AMT2': '3102', 'BILL_AMT3': '689', 'BILL_AMT4': '0',
        'BILL_AMT5': '0', 'BILL_AMT6': '0',
        'PAY_AMT1': '0', 'PAY_AMT2': '689', 'PAY_AMT3': '0', 'PAY_AMT4': '0', 'PAY_AMT5': '0', 'PAY_AMT6': '0'
    }
    res_post_predict = client.post("/predictdata", data=post_payload)
    assert res_post_predict.status_code == 200, f"POST predictdata returned {res_post_predict.status_code}"
    assert b"likely to" in res_post_predict.data, "Prediction text not in response"
    print("[TEST 2 SUCCESS] Flask Web App endpoints (GET /, GET /predictdata, POST /predictdata) working perfectly!")
except Exception as e:
    print(f"[TEST 2 FAILED] Web App Error: {e}")
    sys.exit(1)
