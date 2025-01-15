import pandas as pd
import numpy as np
import onnxruntime as rt
from sklearn.metrics import accuracy_score

from .metamorphic_testing import metamorphic_test_model_2

def model_testing_experiment(model_path):
    # Load data
    X, y = load_data()

    # Load the model
    new_session = rt.InferenceSession(model_path)

    # Run the model
    y_pred = new_session.run(None, {'X': X.values.astype(np.float32)})

    # TODO: add check for whether its model 1 or 2
    metamorphic_test_model_2(X, new_session)

    # Perform tests
    test_summary = {}

    accuracy = accuracy_score(y, y_pred[0])
    test_summary['accuracy'] = accuracy



    # TODO: Add more tests here
    # I think its nice if the test results are added to the test_summary dict that we later upload to a directory "testing_results"
    # save_test_results(test_summary, model_name)

    return test_summary


def load_data():
    data = pd.read_csv('data/investigation_train_large_checked.csv')
    data = data.astype(np.float32)
    data = data.drop(["Ja", "Nee"], axis=1)

    X = data.drop('checked', axis=1)
    y = data['checked']

    return X, y

def save_test_results(test_results, model_name):
    # TODO: Implement this function
    pass

