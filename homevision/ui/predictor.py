import sys

import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

from homevision.features.build_features_v1 import apply_features, read_data
from homevision.models.load import load_model, load_processors

st.title("Model Serving with Streamlit")

run_id = sys.argv[1] if len(sys.argv) > 1 else "b94cd32f170847c1bf05fbc7cc70da2e"

model = load_model(run_id)
processors = load_processors(run_id)
initial_data = read_data()
data = apply_features(initial_data, processors)
data = data[data.ClosePrice.isna()]

X, y = (
    data.loc[:, ~data.columns.isin(["ClosePrice"])],
    data.loc[:, data.columns.isin(["ClosePrice"])],
)

y_pred = model.predict(X)

with st.expander("Data from CSV"):
    st.write(
        "The following dataset has the prediction for rows where ClosePrice is null"
    )
    st.dataframe(X.assign(PREDICTION=y_pred))


with st.expander("Play with the Model!"):
    st.write(
        """
        The form is the way that we can interact with the deployed model.
        Each input is treat as a feature from the dataset. \n
        After submit, the data is processed and the model predicts a value.\n
        If ClosePrice has filled (>0), it will show the related metrics (rmse and mape)
    """
    )
    with st.form("my_form"):
        st.write("Inside the form")

        baths_total = st.slider("BathsTotal", min_value=0.0, max_value=7.0, step=0.1)
        beds_total = st.slider(
            "BedsTotal",
            min_value=int(initial_data.BedsTotal.min()),
            max_value=int(initial_data.BedsTotal.max()),
            step=1,
        )
        cdom = st.slider(
            "CDOM",
            min_value=0,
            max_value=int(initial_data.CDOM.max()),
        )
        lotsize_area = st.slider(
            "LotSizeAreaSQFT",
            min_value=float(initial_data.LotSizeAreaSQFT.min()),
            max_value=float(initial_data.LotSizeAreaSQFT.max()),
        )
        sqft_total = st.slider(
            "SqFtTotal",
            min_value=int(initial_data.SqFtTotal.min()),
            max_value=int(initial_data.SqFtTotal.max()),
            step=1,
        )
        elementary_school_name = st.selectbox(
            "ElementarySchoolName", initial_data.ElementarySchoolName.unique()
        )

        close_price = st.number_input(
            "ClosePrice",
        )
        st.write("NOTE: if ClosePrice != 0, it calculates a rmse and mape")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            submitted = False
            user_input = pd.DataFrame.from_records(
                [
                    (
                        str(baths_total),
                        beds_total,
                        cdom,
                        lotsize_area,
                        int(sqft_total),
                        elementary_school_name,
                        close_price,
                    )
                ],
                columns=[
                    "BathsTotal",
                    "BedsTotal",
                    "CDOM",
                    "LotSizeAreaSQFT",
                    "SqFtTotal",
                    "ElementarySchoolName",
                    "ClosePrice",
                ],
            )

            user_input = apply_features(user_input, processors)
            user_X = user_input.loc[:, ~data.columns.isin(["ClosePrice"])]
            user_y = user_input.loc[:, data.columns.isin(["ClosePrice"])]

            user_y_pred = model.predict(user_X)

            st.write("PREDICTED ClosePrice: ", user_y_pred[0])

            if close_price > 0:
                st.write(
                    "RMSE: ",
                    mse([close_price], user_y_pred, squared=False),
                    " - MAPE:",
                    mape([close_price], user_y_pred),
                )
