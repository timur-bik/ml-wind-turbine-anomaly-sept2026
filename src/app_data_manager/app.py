import os
import sys
from pathlib import Path

import pandas as pd
from data_manager import DataManager  # type: ignore
from utils import read_config  # type: ignore

# Add project root and app_ui directory to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
os.chdir(project_root)


if __name__ == "__main__":
    config = read_config(os.path.join(project_root, "conf", "base", "parameters.yml"))
    data_manager = DataManager(config["data_manager"])
    # data_manager.init_raw_db_table()
    # data = data_manager.get_last_n_points(10, "raw_data")
    # # print("Last 10 rows of raw data:", data)
    # data_manager.init_predictions_db_table()
    # data_manager.init_errors_db_table()
    # data_manager.init_anomalies_db_table()

    # inference_data = pd.read_parquet(
    #     os.path.join(project_root, "data", "01_raw", "df_prod.parquet")
    # )
    # data_manager.insert_data_to_db(inference_data[:5], table_name="raw_data")
    # # print("Last 10 rows of raw data after insertion:", data_manager.get_last_n_points(10, "raw_data"))
    # data = data_manager.get_data_since_timestamp(
    #     start_timestamp="2008-10-01 00:00:00", 
    #     table_name="raw_data"
    #     )
    # # print(data[:5])
    # # print(data[-5:])
    # data = data_manager.get_data_by_timestamp_range(
    #     start_timestamp="2008-10-01 00:00:00",
    #     end_timestamp="2008-10-02 00:00:00",
    #     table_name="raw_data"
    # )
    # print(data)
    data_predictions = data_manager.get_last_n_points(10, table_name="predictions")
    print(data_predictions)
    data_errors = data_manager.get_last_n_points(10, table_name="errors")
    print(data_errors)
    data_anomalies = data_manager.get_last_n_points(10, table_name="anomalies")
    print(data_anomalies)