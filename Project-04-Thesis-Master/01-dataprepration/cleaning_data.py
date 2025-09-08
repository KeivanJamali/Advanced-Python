import numpy as np
import pandas as pd
from pathlib import Path

class Cleaner:
    def __init__(self, 
                 data_people: pd.DataFrame, 
                 data_family: pd.DataFrame, 
                 data_trips: pd.DataFrame) -> None:
        self.people_data = data_people
        self.family_data = data_family
        self.trips_data = data_trips

    def set_setting(self):
        # Population data
        self.population_data = self.people_data.merge(self.family_data,
                                                      on=["questionnaire_code"],
                                                      how="left")
        self.population_data = self.population_data.drop(columns=['home_area_code_y', 'home_region_code_y'], errors='ignore')
        column_updated = " ".join(self.population_data.columns).replace("home_area_code_x", "home_area_code").replace("home_region_code_x", "home_region_code").split(" ")
        self.population_data.columns = column_updated
        self.population_data["family_id"] = self.population_data["questionnaire_code"]
        self.population_data.drop("questionnaire_code", axis=1, inplace=True)

        # Activity chain
        self.activity_chain = self.trips_data.copy()
        self.activity_chain["start_time"] = self.activity_chain["start_hour"] * 60 + self.activity_chain["start_minute"]
        self.activity_chain["family_id"] = self.activity_chain["questionnaire_code"]
        self.activity_chain.drop("questionnaire_code", axis=1, inplace=True)
        self.activity_chain.sort_values(by=["family_id", "person_number_in_family", "start_time"])

        grouped_data = self.activity_chain.groupby(["family_id", "person_number_in_family"])
        records = []
        for (fid, pid), person_df in grouped_data:
            person_df = person_df.sort_values("start_time")
            activity_chain = person_df["trip_code"].tolist()
            time_chain = person_df["start_time"].tolist()
            purpose_chain = person_df["trip_purpose"].tolist()
            duration = [j - i for i, j in zip(time_chain[:-1], time_chain[1:])]
            area_zones = person_df["origin_area_code"].tolist() + [person_df["destination_area_code"].tolist()[-1]]
            region_zones = person_df["origin_region_code"].tolist() + [person_df["destination_region_code"].tolist()[-1]]
            travel_mode = person_df["travel_mode"].tolist()
            trip_distance = person_df["trip_distance"].tolist()
            hourly_correction_factor = person_df["hourly_correction_factor"].tolist()
            records.append({"family_id": fid,
                            "person_number_in_family": pid,
                            "activity_chain": activity_chain,
                            "time_chain": time_chain,
                            "purpose_chain": purpose_chain,
                            "duration": duration,
                            "area_zones": area_zones,
                            "region_zones": region_zones,
                            "travel_mode": travel_mode,
                            "trip_distance": trip_distance,
                            "hourly_correction_factor": hourly_correction_factor})
        self.activity_chain = pd.DataFrame(records)

        self.activity_chain = self.activity_chain.merge(self.population_data,
                                                        on=["family_id", "person_number_in_family"],
                                                        how="left")
        
        self.activity_chain["ends_in_home"] = (self.activity_chain["purpose_chain"].apply(lambda x: x[-1]) == "return_home").astype(float)

    def compute_od_matrix(self, weighted=True):
        df = self.trips_data.copy()
        
        # Use weight or just count
        weight = df["hourly_correction_factor"] if weighted else 1
        
        # Group and sum
        od_matrix_region = (
            df.groupby(["origin_region_code", "destination_region_code"])
            .apply(lambda x: weight[x.index].sum())
            .unstack(fill_value=0)
            .astype(int)
        )
        od_matrix_area = (
            df.groupby(["origin_area_code", "destination_area_code"])
            .apply(lambda x: weight[x.index].sum())
            .unstack(fill_value=0)
            .astype(int)
        )
        
        self.od_matrix_area = od_matrix_area
        self.od_matrix_region = od_matrix_region

        
    def save_data(self, path):
        self.activity_chain.to_csv(Path(path)/"activity_chain.csv")
        self.od_matrix_area.to_csv(Path(path)/"OD_matrix_area.csv")
        self.od_matrix_region.to_csv(Path(path)/"OD_matrix_region.csv")
        self.trips_data.to_csv(Path(path)/"trips.csv")