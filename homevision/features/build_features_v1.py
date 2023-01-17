from typing import TypedDict

import pandas as pd
import pandera as pa
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from homevision.config.environment import Environment as env
from homevision.schemas.raw_schema import RawSchema

# TODO: add a decorator enforcing the copy of dataframe, to avoid side-effects


class ProcessorsV1(TypedDict):
    CDOM: list[StandardScaler]
    SqFtTotal: list[StandardScaler]
    ElementarySchoolName: list[OneHotEncoder]


@pa.check_types
def get_data(file_path: str) -> pa.typing.DataFrame[RawSchema]:
    data = pd.read_csv(file_path, dtype=dict(BathsTotal=str))

    return pa.typing.DataFrame[RawSchema](
        data[
            [
                "BathsTotal",
                "BedsTotal",
                "CDOM",
                "LotSizeAreaSQFT",
                "SqFtTotal",
                "ElementarySchoolName",
                "ClosePrice",
            ]
        ]
    )


@pa.check_types
def clean_data(df: pa.typing.DataFrame[RawSchema]) -> pa.typing.DataFrame[RawSchema]:
    # to avoid side-effects
    data = df.copy()

    data.LotSizeAreaSQFT = data.LotSizeAreaSQFT.fillna(0)

    # =========================
    data.loc[:, ["ElementarySchoolName"]] = data.ElementarySchoolName.replace(
        " ",
        "_",
        regex=True,
    ).str.lower()

    # =========================

    # make positive CDOM values
    data.loc[data.CDOM < 0, ["CDOM"]] = data[data.CDOM < 0][["CDOM"]] * -1

    return data


def build_processors(
    data: pa.typing.DataFrame[RawSchema],
) -> tuple[pd.DataFrame, ProcessorsV1]:

    # ========== CDOM =============
    cdom_scaler = StandardScaler()
    cdom_scaler.fit(data[["CDOM"]])

    # ========== SqFtTotal =============
    sqft_total_scaler = StandardScaler()
    sqft_total_scaler.fit(data[["SqFtTotal"]])

    # one-hot ElementarySchoolName and add default category for ElementarySchoolName (not listed)
    #   - for categories cardinality less than 5(?), add it to the default column
    encoder = OneHotEncoder(
        min_frequency=5,
        handle_unknown="infrequent_if_exist",
    )

    encoder.fit(data[["ElementarySchoolName"]])

    return data, ProcessorsV1(
        CDOM=[cdom_scaler],
        SqFtTotal=[sqft_total_scaler],
        ElementarySchoolName=[encoder],
    )


def apply_processors(df: pa.typing.DataFrame[RawSchema], processors: ProcessorsV1):
    """Apply processors, they should be pretrained

    Args:
        data (pd.DataFrame)
        processors (ProcessorsV1):

    Returns:
        data (pd.DataFrame): information modified
    """
    # TODO: add DataFrame output
    # to avoid side-effects
    data = df.copy()

    # split BathsTotal in BathFull and BathHalf
    data["BathsFull"] = (
        data.BathsTotal.astype(str).str.split(".").str[0].fillna(0).astype(int)
    )
    data["BathsHalf"] = (
        data.BathsTotal.astype(str).str.split(".").str[1].fillna(0).astype(int)
    )

    # update BathsTotal to be BathFull + BathsHalf
    data["BathsTotal"] = data[["BathsFull", "BathsHalf"]].sum(axis=1)

    encoder = processors.get("ElementarySchoolName")[0]
    if isinstance(encoder, OneHotEncoder):
        features_ElementarySchoolName = encoder.get_feature_names_out(
            ["ElementarySchoolName"]
        )
        data[features_ElementarySchoolName] = encoder.transform(
            data[["ElementarySchoolName"]]
        ).toarray()  # type: ignore

        data.drop(columns=data[["ElementarySchoolName"]], inplace=True)

    # - ratio CDOM/sqft
    data["CDOM/SqFtTotal"] = data["CDOM"].divide(data["SqFtTotal"])

    # - add ratio LotSizeAreaSQFT/SqFtTotal
    data["LotSizeAreaSQFT/SqFtTotal"] = data["LotSizeAreaSQFT"].divide(
        data["SqFtTotal"]
    )

    # - add column "SqFtTotal - LotSizeAreaSQFT" '>' and  '<='
    data["LotSizeAreaSQFT>SqFtTotal"] = (
        data["LotSizeAreaSQFT"] > data["SqFtTotal"]
    ).astype(int)

    data["LotSizeAreaSQFT>!SqFtTotal"] = (
        data["LotSizeAreaSQFT"] <= data["SqFtTotal"]
    ).astype(int)

    scaler = processors.get("SqFtTotal")[0]
    if isinstance(scaler, StandardScaler):
        data["SqFtTotal"] = scaler.transform(data[["SqFtTotal"]])

    scaler = processors.get("CDOM")[0]
    if isinstance(scaler, StandardScaler):
        data["CDOM"] = scaler.transform(data[["CDOM"]])

    return data


def build_features(df: pa.typing.DataFrame[RawSchema]):
    data = pa.typing.DataFrame[RawSchema](df.copy())

    # can't use evaluation data in this instance
    # TODO: add a check to assert it
    data = data[~data.ClosePrice.isna()]
    data = clean_data(pa.typing.DataFrame[RawSchema](data))
    _, processors = build_processors(data)

    data = apply_processors(data, processors)

    return data, processors


@pa.check_types
def apply_features(df: pa.typing.DataFrame[RawSchema], processors: ProcessorsV1):
    data: pa.typing.DataFrame[RawSchema] = pa.typing.DataFrame[RawSchema](df.copy())

    data = clean_data(data)
    data = apply_processors(data, processors)
    return data


@pa.check_types
def read_data() -> pa.typing.DataFrame[RawSchema]:
    data = get_data(f"{env.FILE_PATH}/raw/home-listings-example.csv")

    return data
