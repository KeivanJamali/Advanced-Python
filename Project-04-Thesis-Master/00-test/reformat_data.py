import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def people_reformat(data_path:str):
    data = pd.read_excel(Path(data_path))
    data.columns = ["questionnaire_code", "person_number", "gender", "age", "job", "working", "driving_license", "education_level", "home_area_code", "home_region_code"]

    data["gender"] = data["gender"].map({"مرد": "male",
                                   "زن": "female"})
    
    data["job"] = data["job"].map({"خانه دار": "housekeeper",
                                   "فرهنگی": "craftsman",
                                   "دانش آموز": "school_student",
                                   "بیکار": "unemployed",
                                   "کارمند": "employee",
                                   "بازنشسته": "retired",
                                   "پزشک": "doctor",
                                   "سایر": "other",
                                   "کارگر": "worker",
                                   "نظامی": "military",
                                   "دانشجو": "university_student",
                                   "بیکار زیر 18 سال": "unemployed_under_18",
                                   "راننده": "driver",
                                   "استادکار": "teacher",
                                   "فروشنده": "seller"})
    
    data["working"] = data["working"].map({"غیرشاغل": 0,
                                           "شاغل": 1})
    
    data["driving_license"] = data["driving_license"].map({"دارد": 1,
                                                           "ندارد": 0,
                                                           "نامشخص": "n"})

    data["education_level"] = data["education_level"].map({"بی سواد": 0,
                                                           "ابتدایی": 1,
                                                           "راهنمایی": 2,
                                                           "دبیرستان": 3,
                                                           "دیپلم": 4,
                                                           "فوق دیپلم": 5,
                                                           "لیسانس": 6,
                                                           "فوق لیسانس": 7,
                                                           "دکتری و بالاتر": 7})
    
    return data


def family_reformat(data_path:str):
    data = pd.read_excel(Path(data_path))
    data.columns = ["questionnaire_code", "family_members_count", "total_vehicles_count", "bicycles_24inch_or_larger_count", "motorcycles_count", "private_cars_count", "pickup_trucks_count", "taxis_count", "other_vehicles_count", "private_car_1", "private_car_2", "private_car_3", "private_car_4", "home_area_code", "home_region_code", "correction_factor", "expansion_factor"]
    
    return data

def trips_reformat(data_path:str):
    data = pd.read_excel(Path(data_path))
    data.columns = ["trip_code", "questionnaire_code", "person_number", "home_area_code", "home_region_code", "origin_area_code", "origin_region_code", "destination_area_code", "destination_region_code", "trip_purpose", "travel_mode", "start_hour", "start_minute", "hourly_correction_factor", "trip_distance", "home_based", "previous_trip_code"]

    data["trip_purpose"] = data["trip_purpose"].map({'مراجعه به ادارات': "visit_offices",
                                                     'همراهی با دیگران': "accompany_others",
                                                     'رساندن یا اوردن دیگران': "dropoff_or_pickup_others",
                                                     'کار': "work",
                                                     'دیدار نزدیکان': "visit_relatives",
                                                     'خرید': "shopping",
                                                     'تفریح یا زیارت': "recreation_or_pilgrimage",
                                                     'سایر': "other",
                                                     'موارد پزشکی': "medical_purposes",
                                                     'تحصیلی': "education",
                                                     'بازگشت به منزل': "return_home"})

    data["travel_mode"] = data["travel_mode"].map({"سواری شخصی": "private_car",
                                                   "تاکسی": "taxi",
                                                   "اتوبوس واحد": "city_bus",
                                                   "سرویس - سواری": "service_sedan",
                                                   "دوچرخه": "bicycle",
                                                   "تاکسی تلفنی": "phone_taxi",
                                                   "سرویس - مینی بوس": "service_minibus",
                                                   "موتور": "motorcycle",
                                                   "وانت": "pickup_truck",
                                                   "مسافرکش شخصی": "private_passenger",
                                                   "مینی بوس": "minibus",
                                                   "سایر": "other",
                                                   "سرویس - اتوبوس": "service_bus",
                                                   "ون": "van",
                                                   "اتوبوس غیرواحد": "intercity_bus",
                                                   "سرویس - ون": "service_van", 
                                                   "مترو": "metro",
                                                   "کامیون": "truck",
                                                   "nan": "n"})
    
    data["home_based"] = data["home_based"].map({"خانه ابتدا": "home_start",
                                                 "خانه انتها": "home_end",
                                                 "هیچ سر خانه": "no_home_end"})
    
    return data