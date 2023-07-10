from unittest.mock import MagicMock, patch, Mock
from import_data import get_demand, get_vre, get_solar_actual, get_solar_estimate, get_gen, get_mcp, aggregate_hourly, drop_incomplete_datapoints, get_data, read_processed_files
import pandas as pd
import numpy as np
import os

def test_get_demand():
    with patch('import_data.EntsoePandasClient') as MockClient:
        # create dummy dataframe
        date_range = pd.date_range(start='2022-04-03 00:00:00+02:00', end='2022-04-30 23:45:00+02:00', freq='15min')
        np.random.seed(42)
        df = pd.DataFrame({'forecasted_load [MWh]': np.random.uniform(low=10, high=100000, size=len(date_range))
                           }, index=date_range)

        # Mock the query_load_forecast method of EntsoePandasClient, Generate the date-time index
        MockClient.return_value.query_load_forecast.return_value = df

        # Use predefined start and end times for the test
        start = pd.Timestamp(year=2022, month=4, day=1, tz="europe/brussels")
        end = pd.Timestamp(year=2022, month=5, day=1, tz="europe/brussels")

        # call funciton
        df_load_prog, df_load_prog_scaled = get_demand(start, end)

        # Verify dataframe properties
        assert isinstance(df_load_prog, pd.DataFrame)
        assert isinstance(df_load_prog_scaled, pd.DataFrame)

        # Check that both dataframes have the correct columns
        expected_columns = ['forecasted_load [MWh]']
        pd.testing.assert_index_equal(df_load_prog.columns, pd.Index(expected_columns))
        pd.testing.assert_index_equal(df_load_prog_scaled.columns, pd.Index(expected_columns))

        # Check that all values in the scaled dataframe are between 0 and 1
        assert df_load_prog_scaled['forecasted_load [MWh]'].between(0, 1).all()

        # Verify that the dataframes are not empty.
        assert not df_load_prog.empty
        assert not df_load_prog_scaled.empty


def test_get_vre():
    with patch('import_data.EntsoePandasClient') as MockClient:
        # create dummy dataframe
        date_range = pd.date_range(start='2022-04-03 00:00:00+02:00', end='2022-04-30 23:45:00+02:00', freq='15min')
        np.random.seed(42)
        df = pd.DataFrame({'Forecasted Solar [MWh]': np.random.uniform(low=10, high=100000, size=len(date_range)),
                           'Forecasted Wind Offshore [MWh]': np.random.uniform(low=10, high=100000, size=len(date_range)),
                           'Forecasted Wind Onshore [MWh]': np.random.uniform(low=10, high=100000, size=len(date_range)),
                           }, index=date_range)

        # Mock the query_load_forecast method of EntsoePandasClient, Generate the date-time index
        MockClient.return_value.query_wind_and_solar_forecast.return_value = df

        # Use predefined start and end times for the test
        start = pd.Timestamp(year=2022, month=4, day=3, tz="europe/brussels")
        end = pd.Timestamp(year=2022, month=4, day=30, tz="europe/brussels")

        # Call the function with the test inputs
        df_re_prog, df_re_prog_scaled = get_vre(start, end)

        # Check that both outputs are dataframes
        assert isinstance(df_re_prog, pd.DataFrame)
        assert isinstance(df_re_prog_scaled, pd.DataFrame)

        # Check that both dataframes have the correct columns
        expected_columns = ['Forecasted Solar [MWh]', 'Forecasted Wind Offshore [MWh]', 'Forecasted Wind Onshore [MWh]']
        pd.testing.assert_index_equal(df_re_prog.columns, pd.Index(expected_columns))
        pd.testing.assert_index_equal(df_re_prog_scaled.columns, pd.Index(expected_columns))

        # Check that all values in the scaled dataframe are between 0 and 1
        assert df_re_prog_scaled['Forecasted Solar [MWh]'].between(0, 1).all()
        assert df_re_prog_scaled['Forecasted Wind Offshore [MWh]'].between(0, 1).all()
        assert df_re_prog_scaled['Forecasted Wind Onshore [MWh]'].between(0, 1).all()

        # Verify that the dataframes are not empty.
        assert not df_re_prog.empty
        assert not df_re_prog_scaled.empty


def test_get_solar_actual():
    with patch('import_data.EntsoePandasClient') as MockClient:
        # create dummy dataframe
        date_range = pd.date_range(start='2022-04-03 00:00:00+02:00', end='2022-04-30 23:45:00+02:00',
                                   freq='15min')

        # Set a seed for reproducibility
        np.random.seed(42)

        # Create the DataFrame
        df = pd.DataFrame(
            data=np.random.uniform(low=0, high=100000, size=len(date_range)),
            index=date_range,
            columns=pd.MultiIndex.from_tuples([('Solar', 'Actual Aggregated')])
        )

        # Mock the query_load_forecast method of EntsoePandasClient, Generate the date-time index
        MockClient.return_value.query_generation.return_value = df

        # Use predefined start and end times for the test
        start = pd.Timestamp(year=2022, month=4, day=3, tz="europe/brussels")
        end = pd.Timestamp(year=2022, month=4, day=30, tz="europe/brussels")

        # Call the function with the test inputs
        solar_actual_scaled = get_solar_actual(start, end)

        # Check that both outputs are dataframes
        assert isinstance(solar_actual_scaled, pd.DataFrame)

        # Check that both dataframes have the correct columns
        expected_columns = ['solar_capacity_actual']
        pd.testing.assert_index_equal(solar_actual_scaled.columns, pd.Index(expected_columns))

        # Check that all values in the scaled dataframe are between 0 and 1
        assert solar_actual_scaled['solar_capacity_actual'].between(0, 1).all()

        # Verify that the dataframes are not empty.
        assert not solar_actual_scaled.empty


def test_get_solar_estimate():
    with patch('import_data.EntsoePandasClient') as MockClient:
        # create dummy dataframe
        date_range = pd.date_range(start='2022-04-03 00:00:00+02:00', end='2022-04-30 23:45:00+02:00',
                                   freq='15min')

        # Set a seed for reproducibility
        np.random.seed(42)

        df = pd.DataFrame({'Forecasted Solar [MWh]': np.random.uniform(low=10, high=100000, size=len(date_range)),
                           'Forecasted Wind Offshore [MWh]': np.random.uniform(low=10, high=100000,
                                                                               size=len(date_range)),
                           'Forecasted Wind Onshore [MWh]': np.random.uniform(low=10, high=100000,
                                                                              size=len(date_range)),
                           }, index=date_range)

        # Mock the query_load_forecast method of EntsoePandasClient, Generate the date-time index
        MockClient.return_value.query_wind_and_solar_forecast.return_value = df

        # Use predefined start and end times for the test
        start = pd.Timestamp(year=2022, month=4, day=3, tz="europe/brussels")
        end = pd.Timestamp(year=2022, month=4, day=30, tz="europe/brussels")

        # Call the function with the test inputs
        solar_estimate_scaled = get_solar_estimate(start, end)

        # Check that both outputs are dataframes
        assert isinstance(solar_estimate_scaled, pd.DataFrame)

        # Check that both dataframes have the correct columns
        expected_columns = ['solar_capacity_forecast']
        pd.testing.assert_index_equal(solar_estimate_scaled.columns, pd.Index(expected_columns))

        # Check that all values in the scaled dataframe are between 0 and 1
        assert solar_estimate_scaled['solar_capacity_forecast'].between(0, 1).all()

        # Verify that the dataframes are not empty.
        assert not solar_estimate_scaled.empty


def test_get_gen():
    with patch('import_data.EntsoePandasClient') as MockClient:
        # create dummy dataframe
        date_range = pd.date_range(start='2022-04-03 00:00:00+02:00', end='2022-04-30 23:45:00+02:00',
                                   freq='15min')

        # Set a seed for reproducibility
        np.random.seed(42)

        df = pd.DataFrame({'Actual Aggregated': np.random.uniform(low=10, high=100000, size=len(date_range))},
                           index=date_range)

        # Mock the query_load_forecast method of EntsoePandasClient, Generate the date-time index
        MockClient.return_value.query_generation_forecast.return_value = df

        # Use predefined start and end times for the test
        start = pd.Timestamp(year=2022, month=4, day=3, tz="europe/brussels")
        end = pd.Timestamp(year=2022, month=4, day=30, tz="europe/brussels")

        # Call the function with the test inputs
        df_gen_prog, df_gen_prog_scaled = get_gen(start, end)

        # Check that both outputs are dataframes
        assert isinstance(df_gen_prog, pd.DataFrame)
        assert isinstance(df_gen_prog_scaled, pd.DataFrame)

        # Check that both dataframes have the correct columns
        expected_columns = ['Actual Aggregated']
        pd.testing.assert_index_equal(df_gen_prog.columns, pd.Index(expected_columns))
        pd.testing.assert_index_equal(df_gen_prog_scaled.columns, pd.Index(expected_columns))

        # Check that all values in the scaled dataframe are between 0 and 1
        assert df_gen_prog_scaled['Actual Aggregated'].between(0, 1).all()

        # Verify that the dataframes are not empty.
        assert not df_gen_prog.empty
        assert not df_gen_prog_scaled.empty


def test_get_mcp():
    with patch('import_data.EntsoePandasClient') as MockClient:
        # create dummy dataframe
        date_range = pd.date_range(start='2022-04-03 00:00:00+02:00', end='2022-04-30 23:00:00+02:00',
                                   freq='1 h')

        # Set a seed for reproducibility
        np.random.seed(42)

        df = pd.DataFrame({'market_price': np.random.uniform(low=-500, high=500, size=len(date_range))},
                           index=date_range)

        # Mock the query_load_forecast method of EntsoePandasClient, Generate the date-time index
        MockClient.return_value.query_day_ahead_prices.return_value = df

        # Use predefined start and end times for the test
        start = pd.Timestamp(year=2022, month=4, day=3, tz="europe/brussels")
        end = pd.Timestamp(year=2022, month=4, day=30, tz="europe/brussels")

        # Call the function with the test inputs
        df_prices = get_mcp(start, end)

        # Check that both outputs are dataframes
        assert isinstance(df_prices, pd.DataFrame)

        # Check that both dataframes have the correct columns
        expected_columns = ['market_price']
        pd.testing.assert_index_equal(df_prices.columns, pd.Index(expected_columns))

        # Verify that the dataframes are not empty.
        assert not df_prices.empty


def test_aggregate_hourly():

    date_range = pd.date_range(start='2022-04-03 00:00:00+02:00', end='2022-04-30 23:45:00+02:00',
                               freq='15min')

    # Set a seed for reproducibility
    np.random.seed(42)

    # test a single
    df = pd.DataFrame({'prices': np.random.uniform(low=-500, high=500, size=len(date_range))},
                      index=date_range)

    df_hourly = aggregate_hourly(df)

    # Verify that the index is hourly
    assert pd.infer_freq(df_hourly.index) == 'H'

    # another test with a multi index
    df = pd.DataFrame(
        data=np.random.uniform(low=10, high=100000, size=len(date_range)),
        index=date_range,
        columns=pd.MultiIndex.from_tuples([('Solar', 'Actual Aggregated')])
    )

    df_hourly = aggregate_hourly(df)

    # Verify that the index is hourly
    assert pd.infer_freq(df_hourly.index) == 'H'


def test_drop_incomplete_datapoints():

    # create dummy dataframe
    date_range = pd.date_range(start='2022-04-03 00:00:00+02:00', end='2022-04-30 23:00:00+02:00',
                               freq='1 h')

    # Set a seed for reproducibility
    np.random.seed(42)

    df = pd.DataFrame({'market_price': np.random.uniform(low=-500, high=500, size=len(date_range))},
                       index=date_range)

    # Call the function to drop incomplete datapoints
    df_dropped = drop_incomplete_datapoints(df)

    # Assert that the resulting DataFrame has complete datapoints (length of 24 for each date)
    assert all(df_dropped.groupby(df_dropped.index.date).apply(len) == 24)

    # Assert that the original DataFrame remains unchanged
    assert len(df) == len(df_dropped)


    # another test: check that it actually removes all entries from a day, if one is missing

    # create dummy dataframe with 22 entries
    date_range = pd.date_range(start='2022-04-04 00:00:00+02:00', end='2022-04-04 22:00:00+02:00',
                               freq='1 h')

    # Set a seed for reproducibility
    np.random.seed(42)

    df = pd.DataFrame({'market_price': np.random.uniform(low=-500, high=500, size=len(date_range))},
                      index=date_range)

    # Call the function to drop incomplete datapoints
    df_dropped = drop_incomplete_datapoints(df)

    # Assert that the resulting DataFrame has complete datapoints (length of 24 for each date)
    assert all(df_dropped.groupby(df_dropped.index.date).apply(len) == 24)

    # Assert that the original DataFrame remains unchanged
    assert len(df_dropped) == 0


def test_get_data():
    # Arrange
    mock_df = Mock(spec=pd.DataFrame)
    mock_get_demand = Mock(return_value=(mock_df, mock_df))
    mock_get_vre = Mock(return_value=(mock_df, mock_df))
    mock_get_gen = Mock(return_value=(mock_df, mock_df))
    mock_get_mcp = Mock(return_value=mock_df)
    mock_get_solar_actual = Mock(return_value=mock_df)
    mock_get_solar_estimate = Mock(return_value=mock_df)
    mock_print = Mock()
    print('done')
    # We need to patch the functions where they are imported, not where they're defined.
    with patch('import_data.get_demand', mock_get_demand), \
         patch('import_data.get_vre', mock_get_vre), \
         patch('import_data.get_gen', mock_get_gen), \
         patch('import_data.get_mcp', mock_get_mcp), \
         patch('import_data.get_solar_actual', mock_get_solar_actual), \
         patch('import_data.get_solar_estimate', mock_get_solar_estimate), \
         patch('builtins.print', mock_print), \
         patch('os.makedirs') as mock_makedirs:

        # Act
        get_data()

        # Assert
        assert mock_get_demand.called
        assert mock_get_vre.called
        assert mock_get_gen.called
        assert mock_get_mcp.called
        assert mock_get_solar_actual.called
        assert mock_get_solar_estimate.called
        assert mock_print.called
        assert mock_makedirs.called

        assert mock_df.to_pickle.call_count == 9 #  assert that to_pickle is called 9 times (once for each df)

def test_read_processed_files():
    # Arrange
    mock_df = MagicMock(spec=pd.DataFrame)
    paths = ['df_demand_path', 'df_demand_scaled_path', 'df_vre_path', 'df_vre_scaled_path',
             'df_gen_path', 'df_gen_scaled_path', 'df_mcp_path', 'df_solar_cap_actual_path',
             'df_solar_cap_forecast_path']

    with patch('pandas.read_pickle', return_value=mock_df) as mock_read_pickle:
        # Act
        results = read_processed_files()

        # Assert
        assert len(results) == len(paths), "Not all files were read"
        assert all(isinstance(result, pd.DataFrame) for result in results), "All results should be DataFrames"

        for path in paths:
            mock_read_pickle.assert_any_call(path)


