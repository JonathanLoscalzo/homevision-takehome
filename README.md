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
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │



--------


## References

- Project structure based on [https://drivendata.github.io/cookiecutter-data-science/](cookiecutter-data-science/)

## ML Process

This process is a generic one, but it could be different for other scenarios.

- analyze and understand the information
- feature engineering: clean, transform, validate(TODO)
- train some regression algorithms with default hyperparameters
- select the best one (using cross-validation)
- tune it (with optuna or grid/random search)
- serve the best one.


## First steps

```
python -m venv
source venv/bin/activate
pip install -r requirements.txt
```

## Interesting stuffs

- cookiecutter template based
- mlflow tracking
- optuna for tuning
- pre-commit to ensure black and isort

## TODO's
- 

## Enhancements

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