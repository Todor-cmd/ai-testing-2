import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def robustness_test(X, y, session, noise_levels=np.linspace(0, 1, 20), with_plot=False, save_path='robustness_plot.png'):
    """
    Test the robustness of a model by evaluating its performance on noisy data.
    
    Args:
        session: InferenceSession object for running the model.
        X: Clean test data as a pandas DataFrame or numpy array.
        y: True labels for the test data.
        noise_levels: Array of noise levels to evaluate.
        save_path: File path to save the robustness plot.
        
    Returns:
        results: A dictionary containing noise levels, accuracies, and prediction differences.
    """
    X_clean = X.values.astype(np.float32)
    y_clean_pred = session.run(None, {'X': X_clean})[0]
    
    accuracies = []
    prediction_differences = []  # Tracks the proportion of differing predictions
    
    for noise_level in noise_levels:
        # Add Gaussian noise to the data
        X_noisy = X_clean + np.random.normal(0, noise_level, X_clean.shape).astype(np.float32)
        
        # Run the model
        y_noisy_pred = session.run(None, {'X': X_noisy})[0]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_noisy_pred)
        prediction_difference = np.mean(y_clean_pred != y_noisy_pred)  # Proportion of different predictions
        
        # Store metrics
        accuracies.append(accuracy)
        prediction_differences.append(prediction_difference)
        
    if with_plot:
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        # Plot Accuracy vs Noise Level
        plt.subplot(1, 2, 1)
        plt.plot(noise_levels, accuracies, marker='o', label='Accuracy', color='blue')
        plt.title('Accuracy vs Noise Level')
        plt.xlabel('Noise Level (std of Gaussian noise)')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        # Plot Prediction Difference Coefficient vs Noise Level
        plt.subplot(1, 2, 2)
        plt.plot(noise_levels, prediction_differences, marker='o', label='Prediction Difference', color='red')
        plt.title('Prediction Difference vs Noise Level')
        plt.xlabel('Noise Level (std of Gaussian noise)')
        plt.ylabel('Prediction Difference Coefficient')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)  # Save the plot to the specified path
        plt.close()  # Close the plot to free up memory
        
    # Return results
    return {
        'noise_levels': noise_levels.tolist(),
        'accuracies': list(accuracies),
        'prediction_differences': list(prediction_differences),
    }



def robustness_test_two_models(session_1, session_2, X, y, noise_levels=np.linspace(0, 1, 20), with_plot=False, save_path='robustness_plot.png'):
    """
    Test the robustness of two models by evaluating their performance on noisy data.
    
    Args:
        session_1: InferenceSession object for the first model.
        session_2: InferenceSession object for the second model.
        X: Clean test data as a pandas DataFrame or numpy array.
        y: True labels for the test data.
        noise_levels: Array of noise levels to evaluate.
        save_path: File path to save the robustness plot.
        
    Returns:
        results: A dictionary containing noise levels, accuracies, and prediction differences for both models.
    """
    X_clean = X.values.astype(np.float32)
    y_clean_pred_1 = session_1.run(None, {'X': X_clean})[0]
    y_clean_pred_2 = session_2.run(None, {'X': X_clean})[0]
    
    accuracies_1 = []
    accuracies_2 = []
    prediction_differences_1 = []  # Tracks the proportion of differing predictions for model 1
    prediction_differences_2 = []  # Tracks the proportion of differing predictions for model 2
    
    if with_plot:
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        # Plot Accuracy vs Noise Level for both models
        plt.subplot(1, 2, 1)
        plt.plot(noise_levels, accuracies_1, marker='o', label='Model 1 Accuracy', color='blue')
        plt.plot(noise_levels, accuracies_2, marker='x', label='Model 2 Accuracy', color='green')
        plt.title('Accuracy vs Noise Level')
        plt.xlabel('Noise Level (std of Gaussian noise)')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        # Plot Prediction Difference Coefficient vs Noise Level for both models
        plt.subplot(1, 2, 2)
        plt.plot(noise_levels, prediction_differences_1, marker='o', label='Model 1 Prediction Diff.', color='blue')
        plt.plot(noise_levels, prediction_differences_2, marker='x', label='Model 2 Prediction Diff.', color='green')
        plt.title('Prediction Difference vs Noise Level')
        plt.xlabel('Noise Level (std of Gaussian noise)')
        plt.ylabel('Prediction Difference Coefficient')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)  # Save the plot to the specified path
        plt.close()  # Close the plot to free up memory
    
    # Return results with converted numpy arrays to lists
    return {
        'noise_levels': noise_levels.tolist(),
        'accuracies_model_1': list(accuracies_1),
        'accuracies_model_2': list(accuracies_2),
        'prediction_differences_model_1': list(prediction_differences_1),
        'prediction_differences_model_2': list(prediction_differences_2),
    }