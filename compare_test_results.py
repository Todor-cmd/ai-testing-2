from utils.fairness_testing import plot_fairness_results
from utils.robustness_testing import plot_robustness_results
import json

model_1_results = json.load(open("results/model_1/results.json"))
model_2_results = json.load(open("results/model_2/results.json"))

fairness_results = {
    'model_1': model_1_results['fairness_metrics'],
    'model_2': model_2_results['fairness_metrics']
}

# plot_fairness_results(fairness_results)

plot_robustness_results(
    model_1_results['robustness_metrics'],
    model_2_results['robustness_metrics']
)


