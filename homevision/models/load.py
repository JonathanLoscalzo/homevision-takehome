import mlflow
from joblib import load

from homevision.features.build_features_v1 import ProcessorsV1


def load_model(run_id="b94cd32f170847c1bf05fbc7cc70da2e"):
    logged_model = f"runs:/{run_id}/final-model"

    # Load model as a PyFuncModel.
    return mlflow.pyfunc.load_model(logged_model)


def load_processors(run_id="b94cd32f170847c1bf05fbc7cc70da2e") -> ProcessorsV1:
    url_path = mlflow.artifacts.download_artifacts(
        run_id="b94cd32f170847c1bf05fbc7cc70da2e"
    )

    cdom_scaler = load(f"{url_path}/processors/CDOM.joblib")
    sqft_scaler = load(f"{url_path}/processors/SqFtTotal.joblib")
    esn_onehot = load(f"{url_path}/processors/ElementarySchoolName.joblib")

    return ProcessorsV1(
        CDOM=cdom_scaler, SqFtTotal=sqft_scaler, ElementarySchoolName=esn_onehot
    )
