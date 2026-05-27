import cdsapi
import os
from pathlib import Path

AREA = [34.6, 114.9, 29.4, 119.7]

YEAR = ['2020', '2021', '2022', '2023', '2024', '2025']
MONTH = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok = True)
os.chdir("data-test")

DATASET = "reanalysis-era5-land"

VARIABLES1 = [
    '2m_temperature',
    'skin_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_solar_radiation_downwards'
]

VARIABLES2 = [
    'surface_net_solar_radiation',
    'surface_pressure',
    'soil_temperature_level_1',
    'surface_sensible_heat_flux',
    'volumetric_soil_water_layer_1',
    'lake_surface_sensible_heat_flux',
    'lake_mix_layer_temperature'
]

DAY = [
    "01", "02", "03",
    "04", "05", "06",
    "07", "08", "09",
    "10", "11", "12",
    "13", "14", "15",
    "16", "17", "18",
    "19", "20", "21",
    "22", "23", "24",
    "25", "26", "27",
    "28", "29", "30",
    "31"
]

TIME = [
    "00:00", "01:00", "02:00",
    "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00",
    "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00",
    "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00",
    "21:00", "22:00", "23:00"
]

client = cdsapi.Client()

for year in YEAR:
    path = Path(year)
    path.mkdir(exist_ok = True)
    os.chdir(year)
    for month in MONTH:
        path = Path(month)
        path.mkdir(exist_ok = True)
        os.chdir(month)

        path = Path("V1")
        path.mkdir(exist_ok = True)
        os.chdir("V1")
        REQUEST = {
            "variable" : VARIABLES1,
            "year" : year,
            "month" : month,
            "day" : DAY, 
            "time" : TIME, 
            "data_format" : "netcdf", 
            "download_format" : "unarchived",
            "area" : AREA,
        }
        if not os.listdir():
            client.retrieve(DATASET, REQUEST).download()
        os.chdir("..")
        
        path = Path("V2")
        path.mkdir(exist_ok = True)
        os.chdir("V2")
        REQUEST = {
            "variable" : VARIABLES2,
            "year" : year,
            "month" : month,
            "day" : DAY, 
            "time" : TIME, 
            "data_format" : "netcdf", 
            "download_format" : "unarchived",
            "area" : AREA,
        }
        if not os.listdir():
            client.retrieve(DATASET, REQUEST).download()
        os.chdir("..")
        
        os.chdir("..")
    os.chdir("..")
