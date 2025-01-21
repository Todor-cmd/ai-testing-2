import pandas as pd
import numpy as np
import onnxruntime as rt
from sklearn.metrics import accuracy_score

from .metamorphic_testing import metamorphic_test_model_2, metamorphic_test_model_1
from .robustness_testing import robustness_test, robustness_test_two_models

def model_testing_experiment(model_path):
    # Load data
    X, y = load_data()

    # Load the model
    new_session = rt.InferenceSession(model_path)

    #Check which model we're testing 
    model = 0
    if "model_1.onnx" in model_path:
        model = 1
    else:
        if "model_2.onnx" in model_path:
            model = 2

    print(f"model = {model}")
    #TODO some check if model is still 0 something is wrong


    # Run the model
    y_pred = new_session.run(None, {'X': X.values.astype(np.float32)})

    # Perform tests
    test_summary = {}

    # Metamorphic tests
   
    if model == 1:
        print("\n\n\nmodel 1 neighbourhood test")
        metamorphic_test_model_1(X, new_session)
        print("\n\n\nmodel 1 age test")
        metamorphic_test_model_2(X, new_session)

    else:
        print("\n\n\nmodel 2 neighbourhood test")
        metamorphic_test_model_1(X, new_session)
        print("model 2 age test")
        metamorphic_test_model_2(X, new_session)



    # Robustness test
    robustness_result = robustness_test(new_session, X, y, save_path=f'model_{model}_robustness_plot.png')
    test_summary["robustness"] = robustness_result


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

