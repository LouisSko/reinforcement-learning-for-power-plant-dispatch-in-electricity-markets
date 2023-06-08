from entsoe import EntsoePandasClient
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# you need to get a key to access the API
# you can ask for it here https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
# the instructions are given below chapter "2. Authentication and Authorisation"

your_key = '9ab2e188-d454-44be-bce7-ea9dc8863723'
# your_key = '3a72c137-c318-4dd5-ac00-2d3be87966a8'

country_code = 'DE_LU'  # Germany-Luxembourg

client = EntsoePandasClient(api_key=your_key)

# set directory and define the file paths
parent_directory = os.path.join('../..', 'data', 'preprocessed')
df_demand_path = os.path.join(parent_directory, 'df_demand.pkl')
df_demand_scaled_path = os.path.join(parent_directory, 'df_demand_scaled.pkl')
df_vre_path = os.path.join(parent_directory, 'df_vre.pkl')
df_vre_scaled_path = os.path.join(parent_directory, 'df_vre_scaled.pkl')
df_gen_path = os.path.join(parent_directory, 'df_gen.pkl')
df_gen_scaled_path = os.path.join(parent_directory, 'df_gen_scaled.pkl')
df_mcp_path = os.path.join(parent_directory, 'df_mcp.pkl')
df_solar_cap_actual_path = os.path.join(parent_directory, 'df_solar_cap_actual.pkl')
df_solar_cap_forecast_path = os.path.join(parent_directory, 'df_solar_cap_forecast.pkl')

def get_demand(start, end):
    """
    Day-ahead Total Load Forecast

    Quelle:
    https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=CTY&atch=false&dateTime.dateTime=09.05.2023+00:00|CET|DAY&biddingZone.values=CTY|10YLU-CEGEDEL-NQ!CTY|10YLU-CEGEDEL-NQ&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)
    LU+DE muss ausgewählt werden -> dann stimmen die Werte

    Do some data manipulation from export from ENTSO-E, so get ready for the observation space
    TODO Check the validity of ENTSO-E data, like are the renewable generation values alsways below the installed capacity, is the data complete etc.

    Returns:
    Demand df for specified start and end date
    """

    # make api call
    df_load_prog = pd.DataFrame(client.query_load_forecast(country_code, start=start, end=end))

    # rename columns
    df_load_prog.columns = ['forecasted_load [MWh]']

    # 15 min candles -> aggregate hourly
    df_load_prog = aggregate_hourly(df_load_prog)

    # drop days with incomplete number of observations (!=24) per day.
    df_load_prog = drop_incomplete_datapoints(df_load_prog)

    # scale Data
    scaler = MinMaxScaler((0, 1))
    df_load_prog_scaled = pd.DataFrame(scaler.fit_transform(df_load_prog), columns=['forecasted_load [MWh]'])

    # set datetime index
    df_load_prog_scaled.index = df_load_prog.index

    # return unscaled and scaled values as a df
    return df_load_prog, df_load_prog_scaled


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

    # make api call
    df_re_prog = pd.DataFrame(client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None))

    # rename columns
    df_re_prog.columns = ['Forecasted Solar [MWh]', 'Forecasted Wind Offshore [MWh]', 'Forecasted Wind Onshore [MWh]']

    # 15 min candles? -> aggregate hourly
    df_re_prog = aggregate_hourly(df_re_prog)

    # drop days with incomplete number of observations (!=24) per day. -> leap years
    df_re_prog = drop_incomplete_datapoints(df_re_prog)

    # scale data
    scaler = MinMaxScaler((0, 1))
    df_re_prog_scaled = pd.DataFrame(scaler.fit_transform(df_re_prog),
                                     columns=['Forecasted Solar [MWh]', 'Forecasted Wind Offshore [MWh]',
                                              'Forecasted Wind Onshore [MWh]'])

    # set index
    df_re_prog_scaled.index = df_re_prog.index

    return df_re_prog, df_re_prog_scaled


def get_solar_actual(start, end):
    """
    Actuals Generation for Solar

    Quelle
    https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=14.03.2023+00:00|CET|DAYTIMERANGE&dateTime.endDateTime=14.03.2023+00:00|CET|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B01&productionType.values=B02&productionType.values=B03&productionType.values=B04&productionType.values=B05&productionType.values=B06&productionType.values=B07&productionType.values=B08&productionType.values=B09&productionType.values=B10&productionType.values=B11&productionType.values=B12&productionType.values=B13&productionType.values=B14&productionType.values=B20&productionType.values=B15&productionType.values=B16&productionType.values=B17&productionType.values=B18&productionType.values=B19&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)

    Do some data manipulation from export from ENTSO-E, so get ready for the observation space
    TODO Check the validity of ENTSO-E data, like are the renewable generation values always below the installed capacity, is the data complete etc.

    Returns:
    Renewable Infeed array for specified start and end date
    """

    # make api call
    solar_actual = pd.DataFrame(client.query_generation(country_code, start=start, end=end, psr_type=None))

    # select data
    solar_actual = solar_actual['Solar'][['Actual Aggregated']].copy()

    # 15 min candles? -> aggregate hourly
    solar_actual = aggregate_hourly(solar_actual)

    # drop days with incomplete number of observations (!=24) per day. -> leap years
    solar_actual = drop_incomplete_datapoints(solar_actual)

    # Group the data by day and apply scaling to each day's values
    solar_actual['solar_capacity_actual'] = solar_actual.groupby(solar_actual.index.date)[
        'Actual Aggregated'].transform(lambda x: x / x.sum())

    return solar_actual[['solar_capacity_actual']]


def get_solar_estimate(start, end):
    """
    Actuals Generation for Solar

    Quelle
    https://transparency.entsoe.eu/generation/r2/actualGenerationPerProductionType/show?name=&defaultValue=false&viewType=TABLE&areaType=CTA&atch=false&datepicker-day-offset-select-dv-date-from_input=D&dateTime.dateTime=14.03.2023+00:00|CET|DAYTIMERANGE&dateTime.endDateTime=14.03.2023+00:00|CET|DAYTIMERANGE&area.values=CTY|10Y1001A1001A83F!CTA|10YDE-VE-------2&productionType.values=B01&productionType.values=B02&productionType.values=B03&productionType.values=B04&productionType.values=B05&productionType.values=B06&productionType.values=B07&productionType.values=B08&productionType.values=B09&productionType.values=B10&productionType.values=B11&productionType.values=B12&productionType.values=B13&productionType.values=B14&productionType.values=B20&productionType.values=B15&productionType.values=B16&productionType.values=B17&productionType.values=B18&productionType.values=B19&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)

    Do some data manipulation from export from ENTSO-E, so get ready for the observation space
    TODO Check the validity of ENTSO-E data, like are the renewable generation values always below the installed capacity, is the data complete etc.

    Returns:
    Renewable Infeed array for specified start and end date
    """

    # make api call
    df_re_prog = pd.DataFrame(client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None))

    # rename columns
    df_re_prog.columns = ['Forecasted Solar [MWh]', 'Forecasted Wind Offshore [MWh]', 'Forecasted Wind Onshore [MWh]']

    solar_estimate = df_re_prog[['Forecasted Solar [MWh]']].copy()

    # 15 min candles? -> aggregate hourly
    solar_estimate = aggregate_hourly(solar_estimate)

    # drop days with incomplete number of observations (!=24) per day. -> leap years
    solar_estimate = drop_incomplete_datapoints(solar_estimate)

    # Group the data by day and apply scaling to each day's values
    solar_estimate['solar_capacity_forecast'] = solar_estimate.groupby(solar_estimate.index.date)[
        'Forecasted Solar [MWh]'].transform(lambda x: x / x.sum())

    return solar_estimate[['solar_capacity_forecast']]


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
    # make api call
    df_gen_prog = pd.DataFrame(client.query_generation_forecast(country_code, start=start, end=end))

    # 15 min candles? -> aggregate hourly
    df_gen_prog = aggregate_hourly(df_gen_prog)

    # drop days with incomplete number of observations (!=24) per day. -> leap years
    df_gen_prog = drop_incomplete_datapoints(df_gen_prog)

    # scale data
    scaler = MinMaxScaler((0, 1))
    df_gen_prog_scaled = pd.DataFrame(scaler.fit_transform(df_gen_prog), columns=['Actual Aggregated'])

    # set index
    df_gen_prog_scaled.index = df_gen_prog.index

    return df_gen_prog, df_gen_prog_scaled


def get_mcp(start, end):
    """
    Do some data manipulation from export from ENTSO-E, so get ready for the observation space

    Returns:
    Day-Ahead market price array for specified start and end date
    in EUR/MWh
    """
    # make api call
    df_prices = pd.DataFrame(client.query_day_ahead_prices(country_code, start=start, end=end))

    # drop days with incomplete number of observations (!=24) per day. -> leap years
    df_prices = drop_incomplete_datapoints(df_prices)

    return df_prices


def aggregate_hourly(df):
    df = df.groupby([pd.Grouper(freq='1h')]).sum()
    return df


def drop_incomplete_datapoints(df):
    # hier werden unter anderem die Daten für Schaltjahre rausgeworfen
    # TODO: Funktion abändern, sodass Schaltjahre berücksichtigt werden in den Daten
    df = df.groupby(df.index.date).filter(lambda x: len(x) == 24)
    return df


def get_data(start=None, end=None, store=True):
    # set start and end date if not provided
    if start is None:
        start = pd.Timestamp(year=2018, month=1, day=1, tz="europe/brussels")
    if end is None:
        end = pd.Timestamp.now(tz="europe/brussels").floor('D') + pd.Timedelta(days=2)

    # get data
    df_demand, df_demand_scaled = get_demand(start, end)
    print('retrieved demand')
    df_vre, df_vre_scaled = get_vre(start, end)
    print('retrieved solar, wind')
    df_gen, df_gen_scaled = get_gen(start, end)
    print('retrieved gen')
    df_mcp = get_mcp(start, end)
    print('retrieved price')
    df_solar_cap_actual = get_solar_actual(start, end)
    print('retrieved solar capacity actual')
    df_solar_cap_forecast = get_solar_estimate(start, end)
    print('retrieved solar capacity forecast')

    if store:
        # create directory if it does not exist
        os.makedirs(parent_directory, exist_ok=True)

        # Storing DataFrames as CSV files
        df_demand.to_pickle(df_demand_path)
        df_demand_scaled.to_pickle(df_demand_scaled_path)
        df_vre.to_pickle(df_vre_path)
        df_vre_scaled.to_pickle(df_vre_scaled_path)
        df_gen.to_pickle(df_gen_path)
        df_gen_scaled.to_pickle(df_gen_scaled_path)
        df_mcp.to_pickle(df_mcp_path)
        df_solar_cap_actual.to_pickle(df_solar_cap_actual_path)
        df_solar_cap_forecast.to_pickle(df_solar_cap_forecast_path)
        print('saved data')

def read_processed_files():
    # Read the pickle files and load as DataFrames
    df_demand = pd.read_pickle(df_demand_path)
    df_demand_scaled = pd.read_pickle(df_demand_scaled_path)
    df_vre = pd.read_pickle(df_vre_path)
    df_vre_scaled = pd.read_pickle(df_vre_scaled_path)
    df_gen = pd.read_pickle(df_gen_path)
    df_gen_scaled = pd.read_pickle(df_gen_scaled_path)
    df_mcp = pd.read_pickle(df_mcp_path)
    df_solar_cap_actual = pd.read_pickle(df_solar_cap_actual_path)
    df_solar_cap_forecast = pd.read_pickle(df_solar_cap_forecast_path)

    return df_demand, df_demand_scaled, df_vre, df_vre_scaled, df_gen, df_gen_scaled, df_solar_cap_forecast, df_solar_cap_actual, df_mcp


if __name__ == '__main__':
    get_data()
