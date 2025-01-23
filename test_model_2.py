from utils.experiment import model_testing_experiment
import os
import json

model_2_path = "models/model_2.onnx"

results = model_testing_experiment(model_2_path)

# Create results directory if it doesn't exist
os.makedirs("results/model_2", exist_ok=True)

# Save results to file as JSON
with open("results/model_2/results.json", "w") as f:
    json.dump(results, f, indent=4)
