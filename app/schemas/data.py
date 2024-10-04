from pydantic import BaseModel

class DataRow(BaseModel):
    Time: str
    Cabin_No: int
    Idu_Status: str
    Temperature: int
    FanSpeed: str
    Mode: str
    Source: str = "Prediction"
