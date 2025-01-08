import sys
sys.path.append('/home/nyto/python-venv/lib/python3.13/site-packages')
# import gdown
import pytest
from detection_fraude import (
    load_and_preprocess_data,
    split_features_target,
    prepare_train_test_data,
    apply_smote,
    train_model,
    evaluate_model
)

# Google Drive file ID commented out but kept for reference
# file_id = '1ak8-oH0YVpk1ZMl_grndA4dI8NodmS9P'
# url = f'https://drive.google.com/uc?id={file_id}'
# output = 'creditcard.csv'
# gdown.download(url, output, quiet=False)

@pytest.fixture(scope="session")
def trained_model_and_data():
    """Session-scoped fixture that trains model once for all tests"""
    df = load_and_preprocess_data('creditcard.csv')
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = prepare_train_test_data(X, y)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    model = train_model(X_train_resampled, y_train_resampled)
    metrics = evaluate_model(model, X_test, y_test)
    return model, X_test, y_test, metrics

@pytest.fixture
def metrics(trained_model_and_data):
    """Returns pre-computed metrics"""
    _, _, _, metrics = trained_model_and_data
    return metrics

def test_accuracy_threshold(metrics):
    assert metrics['accuracy'] >= 0.9, f"Accuracy {metrics['accuracy']} is below threshold of 0.9"

def test_recall_threshold(metrics):
    assert metrics['recall'] >= 0.8, f"Recall {metrics['recall']} is below threshold of 0.8"

def test_f1_threshold(metrics):
    assert metrics['f1'] >= 0.8, f"F1-score {metrics['f1']} is below threshold of 0.8"

def test_auc_roc_threshold(metrics):
    assert metrics['auc_roc'] >= 0.9, f"AUC-ROC {metrics['auc_roc']} is below threshold of 0.9"
