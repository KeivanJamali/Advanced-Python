import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import Information as info
from ast import literal_eval


class PreTrainDataset(Dataset):
    def __init__(self, 
                 data_activity: pd.DataFrame, 
                 dynamic_features: list, 
                 static_features: list,
                 easy_features: list, 
                 target_col: list,
                 seq_len=10):
        """
        Initializes a new instance of the class.

        Args:
            data (pd.DataFrame): The data to be used for initialization.

        Returns:
            None
        """
        self.data_activity = data_activity
        self.dynamic_features = dynamic_features # For transformer model to extract z
        self.static_features = static_features # For mlp model to extract z
        self.easy_features = easy_features
        self.target_col = target_col
        self.seq_len = seq_len

        self.x1, self.x2, self.x, self.y = self._build_sequences()

    def _build_sequences(self):
        X_list = []
        for _, row in self.data_activity.iterrows():
            seq_data = []
            for feat in self.dynamic_features:
                val = row[feat]
                if isinstance(val, list):
                    seq_data.append(val[:self.seq_len] + [0]*(self.seq_len - len(val)))
                else:
                    seq_data.append([val]*self.seq_len)

            seq_data = np.stack(seq_data, axis=-1)  # shape: [seq_len, num_features]
            X_list.append(seq_data)
        
        static = self.data_activity[self.static_features].values
        easy = self.data_activity[self.easy_features].values
        y_list = self.data_activity[self.target_col].values
        return np.array(X_list, dtype=np.float32), static, easy, y_list

    def __len__(self):
        """You most probably need to change the slices."""
        return len(self.data_activity)

    def __getitem__(self, idx):
        """You most probably need to change the slices. returns x1, x2, x3, y
        x1 will be a sequence used in transformers. x2 will be some static features to help the first training phase. 
        x3 is some features that is easy accessable and will be used in the seccond phase.
        y is the possability of going to some place
        (origin name, origin population, destination name, destination population, distance, [time] -> how possable to go there)"""
        return torch.tensor(self.x1[idx], dtype=torch.float), torch.tensor(self.x2[idx], dtype=torch.float), torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.float)


class FineTuneTrainDataset(Dataset):
    def __init__(self, 
                 data_trips: pd.DataFrame, 
                 easy_features: list, 
                 target_col):
        """
        Initializes a new instance of the class.

        Args:
            data (pd.DataFrame): The data to be used for initialization.

        Returns:
            None
        """
        self.data_trips = data_trips
        self.easy_features = easy_features # For mlp model to extract the travel destination(classification model)
        self.target_col = target_col # possibility

        self.X, self.y = self._build_sequences()

    def _build_sequences(self):
        easy = self.data_trips[self.easy_features].values
        y_list = self.data_trips[self.target_col].values

        return easy, np.array(y_list, dtype=np.int64)

    def __len__(self):
        """You most probably need to change the slices."""
        return len(self.data_trips)

    def __getitem__(self, idx):
        """You most probably need to change the slices. returns x1, x2, x3, y
        x1 will be a sequence used in transformers. x2 will be some static features to help the first training phase. 
        x3 is some features that is easy accessable and will be used in the seccond phase.
        y is the possability of going to some place
        (origin name, origin population, destination name, destination population, distance, [time] -> how possable to go there)"""
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.long)



class MyDataloader:
    def __init__(self, data_list:list, train_percent, val_percent, test_percent, batch_size, seed=42):
        """[activity_chain, true_OD_matrix, trips]"""
        self.raw_data_activity = data_list[0]
        self.raw_data_true_od = data_list[1]
        self.raw_data_trips = data_list[2]
        self.batch_size = batch_size
        self.seed = seed
        self.scalers = {}
        self.set_setting()
        train_data, val_data, test_data = self._split_data(data=self.data_trips, 
                                                            train_percent=train_percent, 
                                                            val_percent=val_percent,
                                                            test_percent=test_percent)
        # train_data, val_data, test_data = self._scale_data(train_data, val_data, test_data)
        self.train_dataset, self.val_dataset, self.test_dataset = self._make_datasets(train_data=train_data,
                                                                                    val_data=val_data,
                                                                                    test_data=test_data)
    
    def set_setting(self):
        data_activity = self.raw_data_activity.copy()

        drop_cols = ['Unnamed: 0', "home_area_code", "trip_distance"]
        data_activity.drop(columns=[c for c in drop_cols if c in data_activity.columns], inplace=True)

        # We saw that the data 14075 has trip mode of [nan, nan]. also data 36942 [nan, nan]. remove theses two rows.
        data_activity.drop([14075, 36942], inplace=True, axis=0)
        data_activity.reset_index(drop=True, inplace=True)
        
        # you can change region_zones with area_zones
        # you can change home_region_code with home_area_code
        list_cols = ["activity_chain", "time_chain", "purpose_chain", "area_zones", "duration", "travel_mode", "trip_distance", "hourly_correction_factor"]
        for col in list_cols:
            if col in data_activity.columns and data_activity[col].dtype == object:
                data_activity[col] = data_activity[col].apply(lambda x: literal_eval(str(x)) if pd.notna(x) else [])

        # Define dynamic (sequence) and static features

        self.data_activity = data_activity

        # ============================================================================================================
        
        # you can change home_region_code with home_area_code
        # you can change destination_region_code with destination_area_code
        data_trips = self.raw_data_trips.copy()
        data_trips["time"] = data_trips["start_hour"] * 60 + data_trips["start_minute"]
        data_trips = data_trips[["home_area_code", "destination_area_code", "trip_distance"]]

        distance_matrix = data_trips.pivot_table(
            index="home_area_code",
            columns="destination_area_code",
            values="trip_distance",
            aggfunc="mean"
        )

        distance_matrix = distance_matrix.fillna(0)

        self.distance_matrix = distance_matrix

        # ============================================================================================================

        data_trips = self.raw_data_trips.copy()
        data_trips["time"] = data_trips["start_hour"] * 60 + data_trips["start_minute"]
        # you can change home_region_code with home_area_code
        # you can change destination_region_code with destination_area_code
        data_trips = data_trips[["questionnaire_code", "person_number_in_family", "home_area_code", "time", "destination_area_code", "trip_distance"]]
        data_trips["possibility"] = 1

        homes = data_trips[["home_area_code"]].drop_duplicates()
        dests = data_trips[["destination_area_code"]].drop_duplicates()
        # times = data_trips["time"].unique()

        all_combinations = pd.DataFrame(list(product(
            homes.itertuples(index=False, name=None),
            dests.itertuples(index=False, name=None),
            # times
        )), columns=["home", "dest"])

        all_combinations[["home_area_code"]] = pd.DataFrame(all_combinations["home"].tolist(), index=all_combinations.index)
        all_combinations[["destination_area_code"]] = pd.DataFrame(all_combinations["dest"].tolist(), index=all_combinations.index)

        all_combinations = all_combinations.drop(columns=["home", "dest"])
        home_features = data_trips[["home_area_code", "questionnaire_code", "person_number_in_family"]].drop_duplicates("home_area_code")
        all_combinations = all_combinations.merge(home_features,
                                                on="home_area_code",
                                                how="left")


        merged = all_combinations.merge(
            data_trips[["home_area_code", "time", "destination_area_code", "trip_distance", "possibility"]],
            # on=["home_region_code", "time", "destination_region_code"],
            on=["home_area_code", "destination_area_code"],
            how="left"
        )
        merged["possibility"] = merged["possibility"].fillna(0)

        # Merge activity info into each trip
        merged_trips = merged.merge(
            data_activity,
            left_on=["questionnaire_code", "person_number_in_family"],
            right_on=["family_id", "person_number_in_family"],
            how="left"
        )

        distance_lookup = self.distance_matrix.stack().to_dict()
        def fill_distance(row):
            if pd.isna(row["trip_distance"]):
                return distance_lookup.get(
                    (row["home_area_code"], row["destination_area_code"]), 
                    0  # fallback if pair not found
                )
            return row["trip_distance"]
        merged_trips["trip_distance"] = merged_trips.apply(fill_distance, axis=1)

        self.data_trips = merged_trips

        self.dynamic_features = ['activity_chain', 'time_chain', 'purpose_chain', 'trip_distance', 'travel_mode']
        self.static_features = [
            'gender', 'age', 'job', 'working', 'driving_license', 'education_level',
            'home_area_code', 'family_members_count', 'total_vehicles_count',
            'bicycles_24inch_or_larger_count', 'motorcycles_count', 'private_cars_count',
            'pickup_trucks_count', 'taxis_count', 'other_vehicles_count'
        ]
        self.easy_features = ["home_area_code", "trip_distance", "destination_area_code"]
        self.target = "possibility"

        # ============================================================================================================

    def fit(self):
        self.train_data, self.val_data, self.test_data = self._make_dataloader(self.train_dataset,
                                                                               self.val_dataset,
                                                                               self.test_dataset)
        return self.train_data, self.val_data, self.test_data

    def _split_data(self, data: pd.DataFrame, 
                    train_percent:float, 
                    val_percent:float, 
                    test_percent:float=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
        """
        Splits the data into training, validation, and testing sets.

        Parameters:
            train_percent (float): The percentage of data to be used for training.
            val_percent (float): The percentage of data to be used for validation.
            test_percent (float): The percentage of data to be used for testing.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and testing
            dataframes.
        """
        train_data, val_test_data = train_test_split(data, train_size=train_percent,
                                                     random_state=self.seed)
        val_data, test_data = train_test_split(val_test_data, train_size=val_percent / (val_percent + test_percent),
                                               random_state=self.seed)
        return train_data, val_data, test_data

    def _scale_data(self, train_data, val_data, test_data):
        """
        Scale numeric/static features
        """
        train_scaled, val_scaled, test_scaled = train_data.copy(), val_data.copy(), test_data.copy()

        # Standard scale continuous variables
        num_cols = ['age', 'trip_distance', 'duration']
        for col in num_cols:
            if col in train_data.columns:
                scaler = StandardScaler()
                train_scaled[col] = scaler.fit_transform(train_data[[col]])
                val_scaled[col] = scaler.transform(val_data[[col]])
                test_scaled[col] = scaler.transform(test_data[[col]])
                self.scalers[col] = scaler

        return train_scaled, val_scaled, test_scaled
    
    # def _scale_data(self, 
    #                 train_data:pd.DataFrame, 
    #                 val_data:pd.DataFrame, 
    #                 test_data:pd.DataFrame, 
    #                 have_test:bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     """
    #     Scale the data data using MinMaxScaler.

    #     Parameters:
    #         train_data (pd.DataFrame): The training data.
    #         val_data (pd.DataFrame): The validation data.
    #         test_data (pd.DataFrame): The test data.

    #     Returns:
    #         tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The scaled training, validation, and test data.
    #     """
    #     self.scaler_x = StandardScaler()
    #     cols = ["age"]
    #     train_data[cols] = self.scaler_x.fit_transform(train_data[cols])
    #     val_data[cols] = self.scaler_x.transform(val_data[cols])
    #     if have_test:
    #         test_data[cols] = self.scaler_x.transform(test_data[cols])
    #     return train_data, val_data, test_data
    
    def _make_datasets(self, 
                       train_data:pd.DataFrame, 
                       val_data:pd.DataFrame, 
                       test_data:pd.DataFrame) -> tuple[Dataset, Dataset, Dataset]:
        """
        Generates the datasets for training, validation, and testing.

        Args:
            train_data (np.ndarray): The training data.
            val_data (np.ndarray): The validation data.
            test_data (np.ndarray): The testing data.

        Returns:
            tuple: A tuple containing the train dataset, val dataset, and test dataset.
        """
        train_dataset = PreTrainDataset(data_activity=train_data,
                                        easy_features=self.easy_features,
                                        dynamic_features=self.dynamic_features,
                                        static_features=self.static_features,
                                        target_col=self.target)                                           
        val_dataset = PreTrainDataset(data_activity=val_data,
                                        easy_features=self.easy_features,
                                        dynamic_features=self.dynamic_features,
                                        static_features=self.static_features,
                                        target_col=self.target)                                           
        test_dataset = PreTrainDataset(data_activity=test_data,
                                        easy_features=self.easy_features,
                                        dynamic_features=self.dynamic_features,
                                        static_features=self.static_features,
                                        target_col=self.target)                                           

        return train_dataset, val_dataset, test_dataset

    def _make_dataloader(self,
                         train_dataset: FineTuneTrainDataset, 
                         val_dataset: FineTuneTrainDataset, 
                         test_dataset: FineTuneTrainDataset) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create dataloaders for the given datasets.

        Parameters:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            test_dataset (Dataset): The testing dataset.

        Returns:
            tuple: A tuple containing train_dataloader, val_dataloader, and test_dataloader.
        """
        # pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_dataloader, val_dataloader, test_dataloader
