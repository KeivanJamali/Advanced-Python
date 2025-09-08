import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import Information as info
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        """
        Initializes a new instance of the class.

        Args:
            data (pd.DataFrame): The data to be used for initialization.

        Returns:
            None
        """
        self.data = data
        self.X = self.data[info.features_dataloader].values
        self.y = self.data[info.target_dataloader].values

    def __len__(self):
        """You most probably need to change the slices."""
        return len(self.data)

    def __getitem__(self, item):
        """You most probably need to change the slices."""
        x = torch.tensor(self.X[item], dtype=torch.float)
        y = torch.tensor(self.y[item], dtype=torch.long)
        return x, y


class MyDataloader:
    def __init__(self, data_list: list, train_percent: float, val_percent: float, test_percent: float, batch_size: int, random_seed: int=42):
        """
        Initializes the object with the specified parameters.

        Parameters:
            data_list (list): [people_data, family_data, trips_data]
            train_percent (float): The percentage of data to use for training.
            val_percent (float): The percentage of data to use for validation.
            batch_size (int): The batch size for training.
        """
        self.batch_size_full = 100000
        self.train_dataloader, self.val_dataloader, self.test_dataloader = None, None, None
        self.raw_data = None
        self.data_list = data_list
        self.seed = random_seed
        self.batch_size = batch_size
        self.data = self._setting()
        train_data, val_data, test_data, have_test = self._split_data(data=self.data, 
                                                                      train_percent=train_percent, 
                                                                      val_percent=val_percent,
                                                                      test_percent=test_percent)
        self.train_data, self.val_data, self.test_data = self._scale_data(train_data, val_data, test_data, have_test)
        self.train_dataset, self.val_dataset, self.test_dataset = self._make_datasets(self.train_data, self.val_data,
                                                                                      self.test_data, have_test)

    def _setting(self) -> pd.DataFrame:
        """
        A function that performs some setting operation.
        """
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

        data["driving_license"] = data["driving_license"].astype(float)

        data = data.loc[:, info.target + info.features]
        # print(len(data))
        data = data.dropna()
        # print(len(data))
        data = pd.get_dummies(data, columns=["trip_purpose", "travel_mode", "home_based", "job"], dtype=float)
        info.features_dataloader = data.columns[1:]
        info.target_dataloader = data.columns[:1]
        return data


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
        if test_percent:
            val_data, test_data = train_test_split(val_test_data, train_size=val_percent / (val_percent + test_percent),
                                                   random_state=self.seed)
            return train_data, val_data, test_data, True
        else:
            return train_data, val_test_data, None, False

    def _scale_data(self, 
                    train_data:pd.DataFrame, 
                    val_data:pd.DataFrame, 
                    test_data:pd.DataFrame, 
                    have_test:bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale the data data using MinMaxScaler.

        Parameters:
            train_data (pd.DataFrame): The training data.
            val_data (pd.DataFrame): The validation data.
            test_data (pd.DataFrame): The test data.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The scaled training, validation, and test data.
        """
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        train_data.loc[:, info.features_dataloader] = self.scaler_x.fit_transform(train_data[info.features_dataloader])
        val_data.loc[:, info.features_dataloader] = self.scaler_x.transform(val_data[info.features_dataloader])
        if have_test:
            test_data.loc[:, info.features_dataloader] = self.scaler_x.transform(test_data[info.features_dataloader])
        return train_data, val_data, test_data

    @staticmethod
    def _make_datasets(train_data:pd.DataFrame, 
                       val_data:pd.DataFrame, 
                       test_data:pd.DataFrame, 
                       have_test:pd.DataFrame) -> tuple[Dataset, Dataset, Dataset]:
        """
        Generates the datasets for training, validation, and testing.

        Args:
            train_data (np.ndarray): The training data.
            val_data (np.ndarray): The validation data.
            test_data (np.ndarray): The testing data.

        Returns:
            tuple: A tuple containing the train dataset, val dataset, and test dataset.
        """
        train_dataset = MyDataset(train_data)
        val_dataset = MyDataset(val_data)
        if have_test:
            test_dataset = MyDataset(test_data)
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, val_dataset, None

    def _make_dataloader(self, train_dataset: MyDataset, 
                         val_dataset: MyDataset, 
                         test_dataset: MyDataset) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create dataloaders for the given datasets.

        Parameters:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            test_dataset (Dataset): The testing dataset.

        Returns:
            tuple: A tuple containing train_dataloader, val_dataloader, and test_dataloader.
        """
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        return train_dataloader, val_dataloader, test_dataloader

    def fit(self):
        """
        Fits the model by creating dataloaders for the train, validation, and test datasets.

        Returns:
            train_dataloader (torch.utils.data.DataLoader): Dataloader for the train dataset.
            val_dataloader (torch.utils.data.DataLoader): Dataloader for the validation dataset.
            test_dataloader (torch.utils.data.DataLoader): Dataloader for the test dataset.
        """
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._make_dataloader(self.train_dataset,
                                                                                                 self.val_dataset,
                                                                                                 self.test_dataset)
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def partial_dependence_plot_data(self, features:dict):
        result = {}
        for feature, values in features.items():
            result[feature] = {}
            for value in values:
                new_data = self.data.copy()
                new_data[feature] = value
                new_data.loc[:, info.features_dataloader] = self.scaler_x.transform(new_data[info.features_dataloader])
                new_data.astype(float)
                new_dataset = MyDataset(new_data)
                new_dataloader = DataLoader(new_dataset, batch_size=self.batch_size_full, shuffle=False, drop_last=False)
                result[feature][value] = new_dataloader
        self.pdp_data = result
        return self.pdp_data
    
    def individual_conditional_expectation_data(self, features:dict):
        result = {}
        for feature, values in features.items():
            result[feature] = {"values":values, "data":{}}
            new_data = self.data.copy()
            for i in new_data.index:
                row = new_data.loc[i]
                row_i = row.copy()
                new_data_i = pd.DataFrame(columns=row.index)
                for value in values:
                    row_i[feature] = value
                    new_data_i.loc[len(new_data_i)] = row_i
                new_data_i.loc[:, info.features_dataloader] = self.scaler_x.transform(new_data_i[info.features_dataloader])

                new_dataset_i = MyDataset(new_data_i)
                new_dataloader_i = DataLoader(new_dataset_i, batch_size=self.batch_size_full, shuffle=False, drop_last=False)
                result[feature]["data"][i] = new_dataloader_i
        self.ice_data = result
        return self.ice_data
    
    def accumulated_local_effects_data(self, feature:str, n:int):
        result_data = [[], [], [feature]]
        max_value, min_value = self.data[feature].max(), self.data[feature].min()
        devision = list(np.linspace(min_value, max_value, n+1))
        new_devision = list(np.linspace(min_value, max_value, n+1))
        for i in range(len(devision)-1):
            mask1 = self.data[feature] >= devision[i]
            mask2 = self.data[feature] <= devision[i+1]
            combined_mask = mask1.astype(int)+mask2.astype(int) == 2
            new_data_min = self.data.copy()[combined_mask]
            new_data_max = self.data.copy()[combined_mask]
            if new_data_min.empty:
                new_devision.remove(devision[i+1])
                print(f"[INFO] Empty Interval {devision[i]} to {devision[i+1]}")
                continue
            new_data_min[feature] = devision[i]
            new_data_max[feature] = devision[i+1]
            new_data_min.loc[:, info.features_dataloader] = self.scaler_x.transform(new_data_min[info.features_dataloader])
            new_data_max.loc[:, info.features_dataloader] = self.scaler_x.transform(new_data_max[info.features_dataloader])
            new_dataset_min = MyDataset(new_data_min)
            new_dataset_max = MyDataset(new_data_max)
            new_dataloader_min = DataLoader(new_dataset_min, batch_size=self.batch_size_full, shuffle=False, drop_last=False)
            new_dataloader_max = DataLoader(new_dataset_max, batch_size=self.batch_size_full, shuffle=False, drop_last=False)
            result_data[0].append(new_devision.pop(0))
            result_data[1].append([new_dataloader_min, new_dataloader_max])
        result_data[0].append(new_devision.pop())
        return result_data
    
    def feature_importance_data(self, feature:str=None, seed=42):
        data = [self.train_data.copy(), self.test_data.copy()]
        result = []
        for i in range(2):
            raw_true_data, second_data = train_test_split(data[i], test_size=0.5, random_state=seed)
            result_data = {}
            if len(raw_true_data) < len(second_data):
                second_data = second_data.iloc[:-1, :]
            elif len(second_data) < len(raw_true_data):
                raw_true_data = raw_true_data.iloc[:-1, :]
            raw_permuted_data = raw_true_data.copy()
            raw_true_data = DataLoader(MyDataset(raw_true_data), batch_size=self.batch_size_full, shuffle=False, drop_last=False)
            result_data["True"] = raw_true_data
            if not feature:
                for feature_ in info.features_dataloader:
                    permuted_data = raw_permuted_data.copy()
                    permuted_data[feature_] = second_data[feature_].values
                    permuted_data = DataLoader(MyDataset(permuted_data), batch_size=self.batch_size_full, shuffle=False, drop_last=False)
                    result_data[feature_] = permuted_data
            else:
                raw_permuted_data[feature] = second_data[feature].values
                raw_permuted_data = DataLoader(MyDataset(raw_permuted_data), batch_size=self.batch_size_full, shuffle=False, drop_last=False)
                result_data[feature] = raw_permuted_data
            result.append(result_data)
        return result[0], result[1]
    
    def feature_intractionn_data(self, features:dict):
        result = {}
        feature_name = []
        value_name = []
        for feature, values in features.items():
            result[feature] = {}
            value_name.append(values)
            feature_name.append(feature)
            for value in values:
                new_data = self.data.copy()
                new_data[feature] = value
                new_data.loc[:, info.features_dataloader] = self.scaler_x.transform(new_data[info.features_dataloader])
                new_data.astype(float)
                new_dataset = MyDataset(new_data)
                new_dataloader = DataLoader(new_dataset, batch_size=self.batch_size_full, shuffle=False, drop_last=False)
                result[feature][value] = new_dataloader

        result[feature_name[0]+"|"+feature_name[1]] = {}
        for value1 in value_name[0]:
            for value2 in value_name[1]:
                new_data = self.data.copy()
                new_data[feature_name[0]] = value1
                new_data[feature_name[1]] = value2
                new_data.loc[:, info.features_dataloader] = self.scaler_x.transform(new_data[info.features_dataloader])
                new_data.astype(float)
                new_dataset = MyDataset(new_data)
                new_dataloader = DataLoader(new_dataset, batch_size=self.batch_size_full, shuffle=False, drop_last=False)
                result[feature_name[0]+"|"+feature_name[1]][str(value1)+","+str(value2)] = new_dataloader

        return result


        