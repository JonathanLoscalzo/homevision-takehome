# HomeVision - Take Home


This is a take home interview for HomeVision that focuses primarily on writing clean code that accomplishes a very practical modeling task. Your challenge is to write a script that meets the requirements.

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── mlruns             <- MLFlow runs folder (local storage)
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── homevision                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── experiments     <- scripts to run custom experiments
    │   │   └── model_v1.py <- experiment v1 - model selection
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features_v1.py
    │   │

--------

## ML Process

This process is a generic one, but it could be different for other scenarios.

- analyze and understand the information (exploratory analysis)
- feature engineering: clean, transform, validate(TODO)
- train some regression algorithms with default hyperparameters
    - select the best one (using cross-validation)
- tune it (with optuna or grid/random search)
- serve it

### Metrics

There are some metrics tracked in mlflow :
- Root Mean Squared Error(rmse): same unit as feature, large errors are punished
- Mean Absolute Error(mae): same scale as feature, errors has the same weight
- R2(r2): is a ratio between the variance explained by the model and the total variance.

I will select `rmse` because the problem needs to punish big errors.

## First steps

```
python -m venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
pip install -e .
```

Execute an experiment

```
# it runs a model selection experiment
python -m homevision.experiments.model_v1
```

View experiments in UI:
```
mlflow ui
```

### Notebooks
- 00_data-analysis.ipynb: exploratory data analysis
- 01_selection-models.ipynb: get training metrics and select a model

## Interesting stuffs

- cookiecutter template based
- mlflow tracking
- optuna for tuning
- pre-commit to ensure black and isort

## TODO's
-

## Enhancements

- use `sklearn-pipeline`
- use [DVC](https://dvc.org/) for data sources
- use mlflow with a remote tracking server
- add validations for data (pandera, great-expectations)
    - to ensure the feature engineering process.
    - e.g. tests about distribution, types, missings, outliers.
- monitoring on serving
    - monitor the model and trigger an alert when data drifts. e.g.: evidently or prometheus+tests
- data gather pipeline, like airflow or similar
- github actions to continuous testing/deliver/training
- use zenml for mlops
- add other lib/project management
    - poetry
    - mlflow project


## References

- Project structure based on [https://drivendata.github.io/cookiecutter-data-science/](cookiecutter-data-science/)
