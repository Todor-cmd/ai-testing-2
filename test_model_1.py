from utils.experiment import model_testing_experiment
import os
import json

model_1_path = "models/model_1.onnx"

results = model_testing_experiment(model_1_path)

# Create results directory if it doesn't exist
os.makedirs("results/model_1", exist_ok=True)

# Save results to file as JSON
with open("results/model_1/results.json", "w") as f:
    json.dump(results, f, indent=4)
