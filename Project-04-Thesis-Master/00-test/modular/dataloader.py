import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class Custom_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data_x = data.loc[:, ["production", "attraction", "cost"]]
        self.data_y = data.loc[:, ["demand"]]
        self.data_x = torch.tensor(data=self.data_x.values, dtype=torch.float32)
        self.data_y = torch.tensor(data=self.data_y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, item):
        x = self.data_x[item, :]
        y = self.data_y[item, :]
        return x, y


class Custom_DataLoader:
    def __init__(self, city_name:str, city_path:str):
        self.city_name = city_name
        self.data_path = Path(city_path)

    def fit(self):
        self._read_data()
        self._merge_data()
        self._split_data()
        self._scale_data()
        self._make_datasets()
        self._make_dataloaders()

    def _read_data(self):
        self.raw_input1 = pd.read_csv(self.data_path/f"{self.city_name}_OD.dat", sep='\t')
        self.raw_input2 = pd.read_csv(self.data_path/"travel_time.csv", index_col=0)
        self.raw_input2.columns = self.raw_input2.columns.astype(int)
        self.raw_production = self.raw_input1.groupby("origin")["demand"].sum()
        self.raw_attraction = self.raw_input1.groupby("dest")["demand"].sum()

    def _merge_data(self):
        self.data = self.raw_input1.copy()
        self.data["production"] = self.data["origin"].map(self.raw_production)
        self.data["attraction"] = self.data["dest"].map(self.raw_attraction)
        self.data["cost"] = self.data.apply(lambda x: self.raw_input2.loc[int(x["origin"]), int(x["dest"])], axis=1)

    def _split_data(self):
        pass

    def _scale_data(self):
        # self.data
        pass

    def _make_datasets(self):
        self.dataset = Custom_Dataset(self.data)

    def _make_dataloaders(self):
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=False)
        


