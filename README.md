# ai-testing-2
A test suite created to test two models as part of DSAIT4015 Software Engineering and Testing for AI Systems course at TU Delft. 

## Set up Environment
Option 1: with conda
```bash
conda env create -f environment.yml 
```
Then 
```bash
conda activate aiTesting2Env
```

Option 2: with pip 
```bash
pip install -r requirements.txt
```

## Running tests
We have two models and therefore two files to run. Each generates results in the 'results/'directory for its respective model. 

```bash
# Run tests for model 1
python test_model_1.py

# Run tests for model 2
python test_model_2.py
```


## Comparing two models
To compare the results of the two models after they are generated, the following file can be run.
```bash
python compare_test_results.py
```

It will save generated plots in 'results/comparison_results/ directory'.





