from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime
import numpy as np

# you need to get a key to access the API
# you can ask for it here https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
# the instructions are given below chapter "2. Authentication and Authorisation"
your_key = '9ab2e188-d454-44be-bce7-ea9dc8863723'
country_code = 'DE_LU'  # Germany-Luxembourg

client = EntsoePandasClient(api_key=your_key)


def get_demand(start, end):
    """
    Day-ahead Total Load Forecast

    Quelle:
    https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=CTY&atch=false&dateTime.dateTime=09.05.2023+00:00|CET|DAY&biddingZone.values=CTY|10YLU-CEGEDEL-NQ!CTY|10YLU-CEGEDEL-NQ&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)
    LU+DE muss ausgewählt werden -> dann stimmen die Werte

    Do some data manipulation from export from ENTSO-E, so get ready for the observation space
    TODO Check the validity of ENTSO-E data, like are the renewable generation values alsways below the installed capacity, is the data complete etc.



    Returns:
    Demand array for specified start and end date

    """

    # load data from ENTSO-E API
    df_load_prog = pd.DataFrame(client.query_load_forecast(country_code, start=start, end=end))

    # YOUR CODE

    return df_load_prog


def get_vre(start, end):
    """
        Day-ahead Generation Forecasts for Wind and Solar

        Quelle
        https://transparency.entsoe.eu/generation/r2/dayAheadGenerationForecastWindAndSolar/show?name=&defaultValue=false&viewType=TABLE&areaType=CTY&atch=false&dateTime.dateTime=09.05.2023+00:00|CET|DAYTIMERANGE&dateTime.endDateTime=09.05.2023+00:00|CET|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTY|10Y1001A1001A83F&productionType.values=B16&productionType.values=B18&productionType.values=B19&processType.values=A18&processType.values=A01&processType.values=A40&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)


        Do some data manipulation from export from ENTSO-E, so get ready for the observation space
        TODO Check the validity of ENTSO-E data, like are the renewable generation values always below the installed capacity, is the data complete etc.


        Returns:
        Renewable Infeed array for specified start and end date

    """
    # load data from ENTSO-E API
    df_re_prog = pd.DataFrame(client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None))

    # YOUR CODE

    return df_re_prog


def get_gen(start, end):
    """
    Day-ahead Aggregated Generation

    Quelle:
    https://transparency.entsoe.eu/generation/r2/dayAheadAggregatedGeneration/show?name=&defaultValue=false&viewType=TABLE&areaType=CTY&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=09.05.2023+00:00|CET|DAYTIMERANGE&dateTime.endDateTime=09.05.2023+00:00|CET|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTY|10Y1001A1001A83F&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)


    Do some data manipulation from export from ENTSO-E, so get ready for the observation space
    TODO Check the validity of ENTSO-E data, like are the renewable generation values alsways below the installed capacity, is the data complete etc.

    Returns:
    Renewable Infeed array for specified start and end date
    """

    df_gen_prog = pd.DataFrame(client.query_generation_forecast(country_code, start=start, end=end))

    # YOUR CODE

    return df_gen_prog


def get_mcp(start, end):
    """
    Do some data manipulation from export from ENTSO-E, so get ready for the observation space

    Returns:
    Day-Ahead market price array for specified start and end date
    in EUR/MWh
    """

    # methods that return Pandas Series
    df_prices = pd.DataFrame(client.query_day_ahead_prices(country_code, start=start, end=end))


    # YOUR CODE

    return df_prices


def get_states_list(start, end):
    """
        
    TODO

    Returns:
    Gives a lists of all datetimes for which we have all data

    """

    # YOUR CODE
    return df_states.index
