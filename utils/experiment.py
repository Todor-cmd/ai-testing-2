import pandas as pd
import numpy as np
import onnxruntime as rt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.model_selection import train_test_split
from utils.fairness_testing import fairness_test
from utils.robustness_testing import robustness_test
from utils.metamorphic_testing import metamorphic_test

def model_testing_experiment(model_path):
    # Load data
    X_test, Y_test = load_test_data()

    # Load the model
    session = rt.InferenceSession(model_path)

    # Run the model
    y_pred = session.run(None, {'X': X_test.values.astype(np.float32)})

    # Perform tests
    test_summary = {}

    

    # test results
    test_summary['performance_metrics'] = performance_test(Y_test, y_pred)

    # Create copies for fairness test
    X_test_fair = X_test.copy()
    Y_test_fair = Y_test.copy()
    test_summary['fairness_metrics'] = fairness_test(X_test_fair, Y_test_fair, session)

    # Create copies for robustness test
    X_test_robust = X_test.copy()
    Y_test_robust = Y_test.copy()
    test_summary['robustness_metrics'] = robustness_test(X_test_robust, Y_test_robust, session)

    # Create copy for metamorphic test
    X_test_meta = X_test.copy()
    test_summary['metamorphic_metrics'] = metamorphic_test(X_test_meta, session)

    return test_summary


def load_test_data():
    data = pd.read_csv('data/investigation_train_large_checked.csv')
    data = data.astype(np.float32)
    data = data.drop(["Ja", "Nee"], axis=1)

    
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    X_test = test_data.drop('checked', axis=1)
    Y_test = test_data['checked']

    return X_test, Y_test


def performance_test(Y_test, y_pred):
    return {
        'accuracy': accuracy_score(Y_test, y_pred[0]),
        'recall': recall_score(Y_test, y_pred[0]), 
        'precision': precision_score(Y_test, y_pred[0]),
        'f1_score': f1_score(Y_test, y_pred[0])
    }
    
    
