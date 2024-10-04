import io
import os
from typing import List, Optional
from fastapi.params import File, Form, Query
import pandas as pd

from fastapi import APIRouter, HTTPException, UploadFile
from schemas.data import DataRow

router = APIRouter()
csv_file = "results_1.csv"


@router.get(
    "/",
    summary="Get data from CSV",
    description="Fetches a specified number of rows from the beginning or end of the CSV file.",
)
async def get_data(
    count: int = Query(1, ge=1, description="Number of rows to retrieve"),
    order: str = Query(
        "last",
        regex="^(first|last)$",
        description="Specify whether to retrieve data from the 'first' or 'last' rows",
    ),
):
    """
    This endpoint retrieves a specified number of rows from a CSV file.

    - **count**: Number of rows to fetch. Must be 1 or higher.
    - **order**: Whether to fetch from the 'first' or 'last' rows of the CSV. Defaults to 'last'.

    **Returns** a list of rows, each represented as a dictionary.
    """
    try:
        if not os.path.exists(csv_file):
            raise HTTPException(status_code=404, detail="CSV file not found")

        df = pd.read_csv(csv_file)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found in the CSV file")

        # Select rows based on 'order' parameter
        if order == "last":
            data = df.tail(count)
        else:  # order == "first"
            data = df.head(count)

        data_dict = data.to_dict(orient="records")
        return data_dict

    except HTTPException as e:
        raise e

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=500, detail="The CSV file is empty or could not be read"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal server error occurred")


@router.post(
    "/",
    summary="Save data to CSV",
    description="Appends data to the CSV file from either JSON data or an uploaded CSV file.",
)
async def save_data(
    json_data: Optional[List[DataRow]] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    This endpoint saves data to the CSV file. It accepts either a JSON payload or an uploaded CSV file.

    - If providing **JSON data**: Send a list of data entries as JSON in the request.
    - If providing a **CSV file**: Upload the CSV file directly with the request.

    **Returns** a message indicating successful data saving.
    """
    try:
        if json_data:
            new_rows_df = pd.DataFrame([row.model_dump() for row in json_data])

        elif file:
            content = await file.read()
            new_rows_df = pd.read_csv(io.StringIO(content.decode("utf-8")))

            expected_columns = [
                "Time",
                "Cabin_No",
                "Idu_Status",
                "Temperature",
                "FanSpeed",
                "Mode",
                "Source",
            ]
            if not all(col in new_rows_df.columns for col in expected_columns):
                raise HTTPException(
                    status_code=400,
                    detail="CSV file is missing one or more required columns",
                )

        else:
            raise HTTPException(status_code=400, detail="No JSON data or file uploaded")

        new_rows_df.to_csv(
            csv_file, mode="a", header=not os.path.exists(csv_file), index=False
        )
        return {"message": "Data saved successfully"}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value Error: {str(ve)}")

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=500, detail="Error occurred while reading the uploaded CSV file"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal server error occurred")
