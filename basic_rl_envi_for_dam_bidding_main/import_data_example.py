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
        Do some data manipulation from export from ENTSO-E, so get ready for the observation space
        TODO Check the validity of ENTSO-E data, like are the renewable generation values alsways below the installed capacity, is the data complete etc. 

        Returns:
        Renewable Infeed array for specified start and end date

    """
    # load data from ENTSO-E API
    df_re_prog = pd.DataFrame(client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None))

    # YOUR CODE

    return df_re_prog


def get_gen(start, end):
    """
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
