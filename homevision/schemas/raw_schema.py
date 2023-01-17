import pandera as pa
from pandera.typing import Series


class RawSchema(pa.SchemaModel):
    BathsTotal: Series[str]
    BedsTotal: Series[int]
    CDOM: Series[int]
    LotSizeAreaSQFT: Series[float] = pa.Field(nullable=True)
    SqFtTotal: Series[int]
    ElementarySchoolName: Series[str]
    ClosePrice: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True
