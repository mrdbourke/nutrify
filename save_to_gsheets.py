import streamlit as st
from typing import List
from google.oauth2 import service_account
from googleapiclient import discovery
import os

# Check to see if developing locally (this changes where images are stored)
if os.environ.get("TEST_NUTRIFY_ENV_VAR"):
    print("*** Using testing Google Sheets database ***")
    SPREADSHEET_ID = (
        "1fdEeFZkr7pNIM-C2vSCe9JVVaUnOL5ZfJ5I8c860VoE"  # test database
    )
else:
    SPREADSHEET_ID = (
        "1CLpDSzJd1mAmG0jHfwGyFG6teRE0ayZrjE8gRVpnrZE"  # prod database
    )

# Authorize the service to add rows to Google Sheets
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],  # read credentials from st.secrets,
    # see: https://docs.streamlit.io/en/stable/tutorial/private_gsheet.html
    scopes=scope,
)
service = discovery.build("sheets", "v4", credentials=credentials)

# The A1 notation of a range to search for a logical table of data.
# Values will be appended after the last row of the table.
RANGE_ = "Sheet1!A:E"

# How the input data should be interpreted.
VALUE_INPUT_OPTION = "USER_ENTERED"

# How the input data should be inserted.
INSERT_DATA_OPTION = "INSERT_ROWS"


def append_values_to_gsheet(values_to_add: List):
    """
    Adds values to gsheet.

    Args:
        values_to_add: values to append to target gsheet, in format of list \
            of list, one item per column, e.g. [[col_1, col_2, col_3]]

    Returns:
        response payload to add to target Gsheet
    """
    # Create the JSON-like format for adding rows to the sheet
    value_range_body = {"majorDimension": "ROWS", "values": values_to_add}

    # Make the request payload and execute it
    request = (
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=SPREADSHEET_ID,
            range=RANGE_,
            valueInputOption=VALUE_INPUT_OPTION,
            insertDataOption=INSERT_DATA_OPTION,
            body=value_range_body,
        )
    )
    response = request.execute()

    # Show the output of what was added
    return response
