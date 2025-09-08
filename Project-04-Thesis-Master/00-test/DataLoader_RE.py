import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import Information as info

class MyDataloader:
    def __init__(self, data_list:list, seed=42):
        """[people_data, family_data, trip_data]"""
        np.random.seed(seed)
        self.seed = seed
        self.data_list = data_list
        self.raw_data = None
        self.data = self.set_setting()
        

    def set_setting(self):
        merged_data = self.data_list[2].merge(self.data_list[0],
                                              on=['questionnaire_code', 'person_number'],
                                              how='left')
        merged_data = merged_data.merge(self.data_list[1],
                                        on=['questionnaire_code'],
                                        how='left')
        merged_data = merged_data.drop(columns=['home_area_code_x', 'home_region_code_x','home_area_code_y', 'home_region_code_y'], errors='ignore')
        self.raw_data = merged_data

        info.features_dataloader, info.target_dataloader = info.features.copy(), info.target.copy()
        data = self.raw_data.copy()
        data["trip_purpose"] = data["trip_purpose"].map({"visit_offices": 0,
                                                         "return_home": 1,
                                                         "recreation_or_pilgrimage": 2,
                                                         "shopping": 3,
                                                         "visit_relatives": 4,
                                                         "education": 5,
                                                         "work": 6,
                                                         "medical_purposes": 7,
                                                         "dropoff_or_pickup_others": 8,
                                                         "other": 9,
                                                         "accompany_others": 10})

        # We have 4 NA data.
        data["travel_mode"] = data["travel_mode"].map({"private_car": 0,
                                                       "taxi": 1,
                                                       "city_bus": 2,
                                                       "service_sedan": 3,
                                                       "bicycle": 4,
                                                       "phone_taxi": 5,
                                                       "service_minibus": 6,
                                                       "motorcycle": 7,
                                                       "pickup_truck": 8,
                                                       "private_passenger": 9,
                                                       "minibus": 10,
                                                       "service_bus": 11,
                                                       "van": 12,
                                                       "intercity_bus": 13,
                                                       "service_van": 14,
                                                       "metro": 15,
                                                       "truck": 16,
                                                       "other": 17})

        data["home_based"] = data["home_based"].map({"home_start": 0,
                                                     "home_end": 1,
                                                     "no_home_end": 2})
        
        # Male = 1, Female = 0
        data["gender"] = data["gender"].map({"male": 1,
                                             "female": 0})

        data["job"] = data["job"].map({"worker": 0,
                                       "military": 1,
                                       "housekeeper": 2,
                                       "school_student": 3,
                                       "employee": 4,
                                       "seller": 5,
                                       "craftsman": 6,
                                       "teacher": 7,
                                       "driver": 8,
                                       "university_student": 9,
                                       "retired": 10,
                                       "unemployed": 11,
                                       "unemployed_under_18": 12,
                                       "doctor": 13,
                                       "other": 14})
        
        data["destination_area_code"] = data["destination_area_code"] - 1
        data["origin_area_code"] = data["origin_area_code"] - 1

        data["driving_license"] = data["driving_license"].astype(float)

        data = data.loc[:, info.target + info.features]
        # print(len(data))
        data = data.dropna()
        # print(len(data))
        data = pd.get_dummies(data, columns=["trip_purpose", "travel_mode", "home_based", "job"], dtype=float)
        info.features_dataloader = data.columns[1:]
        info.target_dataloader = data.columns[:1]
        return data


    def _scale_data(self):
        self.scaler = MinMaxScaler()
        self.data[info.features] = self.scaler.fit_transform(self.data[info.features])