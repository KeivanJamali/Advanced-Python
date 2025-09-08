import pandas as pd
from pathlib import Path

def store_matching_file(file_name):
    data = {"105011": ["پيام يغمائي", "file_10"],
            "105012": ["Tayyebah zareian", "file_11"],
            "105007": ["afsaneh ghasemzadeh", "file_7"],
            "105860": ["zahir faryabi", "file_22"],
            "105015": ["mehdi samimi", "file_20"],
            "105266": ["firoozeh ostovar", "file_18"],
            "105335": ["علیرضا رحیمی", "file_21"],
            "105014": ["محمدرضا مولودي ميبدي", "file_13"],
            "105005": ["FATEMAH LATIF", "file_5"],
            "105949": ["khalil sattar", "file_24"],
            "105018": ["Atieh Mirghanizadehbafghi", "file_16"],
            "105002": ["hamid sadeghi", "file_2"],
            "105019": ["mohammad hosin tashakori", "file_17"],
            "105006": ["majid omidvari", "file_6"],
            "105009": ["abasali dehghani", "file_23"],
            "105434": ["kazem mojarrad", "file_19"],
            "105016": ["akram mostaghelchi", "file_14"],
            "105003": ["Marjan Ganjizade", "file_3"],
            "105017": ["samAneh Malek", "file_15"],
            "105008": ["ali zadmehr", "file_8"],
            "105004": ["fariba hemayati", "file_4"],
            "105013": ["ahmad mokhtari", "file_12"],
            "105001": ["Ali Boloor", "file_1"],
            "105010": ["mahmood moalaei", "file_9"]}

    data = pd.DataFrame(data).T
    data.columns = ["full_name", "reference_id"]
    data.to_csv(file_name)

input_folder = Path(r"/mnt/Data1/Python_Projects/Pure-Python/P5/06-HamiWorks/extra_data")
store_matching_file(file_name=input_folder/"people_index.csv")