import sys
import os

sys.path.append(os.getcwd())

from application import app

from application import app

def test_home_page():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200

def test_predictdata_get():
    client = app.test_client()
    response = client.get("/predictdata")
    assert response.status_code == 200

def test_predictdata_post_with_explanation_and_plots():
    client = app.test_client()
    sample_payload = {
        'LIMIT_BAL': '50000',
        'SEX': '2',
        'EDUCATION': '2',
        'MARRIAGE': '1',
        'AGE': '24',
        'PAY_0': '2',
        'PAY_2': '2',
        'PAY_3': '-1',
        'PAY_4': '-1',
        'PAY_5': '-2',
        'PAY_6': '-2',
        'BILL_AMT1': '3913',
        'BILL_AMT2': '3102',
        'BILL_AMT3': '689',
        'BILL_AMT4': '0',
        'BILL_AMT5': '0',
        'BILL_AMT6': '0',
        'PAY_AMT1': '0',
        'PAY_AMT2': '689',
        'PAY_AMT3': '0',
        'PAY_AMT4': '0',
        'PAY_AMT5': '0',
        'PAY_AMT6': '0'
    }
    response = client.post("/predictdata", data=sample_payload)
    assert response.status_code == 200
    html = response.data.decode('utf-8')
    # Check that predictions, positive and negative reasons, and all proof plots are present
    assert 'The cardholder is likely to DEFAULT' in html or 'The cardholder is NOT likely to default' in html
    assert 'Drivers Increasing Default Risk' in html
    assert 'Drivers Protecting Against Default' in html
    assert 'Proof 1: Feature Impact Breakdown' in html
    assert 'Proof 2: Default Risk Gauge' in html
    assert 'Proof 3: Comparative Benchmark' in html
    assert 'data:image/png;base64,' in html
