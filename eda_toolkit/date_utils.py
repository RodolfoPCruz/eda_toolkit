"""
This module contains functions to parse and format dates
"""

from datetime import datetime


def parse_and_format_dates(
    date_string: str,
    standard_format: str = "%Y-%m-%d",
    return_type: str = "string",
):
    """
    Convert a date string into a datetime object or a formatted string.

    The function tests if the input string matches any of the following
    date formats:

        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d %b %Y",
        "%B %d, %Y"

    If the input string doesn't follow any of these formats, the function will
    return None.
    It can be returned a datetime object or a string.

    Args:
        date_string: date in string format
        standard_format: format of data that will be returned
        return_type: Type of object to return, either 'datetime' or 'string'.

    Returns:
        Union[datetime, str, None]: Parsed date as a datetime object or a
        formatted string. Returns None if parsing fails.

    """

    # accepted
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d %b %Y", "%B %d, %Y"]

    # @if the input is not a string, return None
    if not isinstance(date_string, str):
        return None
    if return_type not in ("string", "datetime"):
        print(
            f"{return_type} is not known. The formats accepted are string "
            "and datetime. String will be used as default"
        )
        return_type = "string"

    for expected_format in formats:
        try:
            # convert to datetime using the format specified in expected_format
            parsed_date = datetime.strptime(date_string, expected_format)
            return (
                parsed_date
                if return_type == "datetime"
                else parsed_date.strftime(standard_format)
            )
        except ValueError:
            continue

    return None
