import torch
import xarray as xr
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = "data"
YEARS = ['2020', '2021', '2022', '2023', '2024', '2025']
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
OUTPUT_DIR = "pt_data"
OUTPUT_FILE = "era5_data.pt"

V1_VARS = ['t2m', 'skt', 'd2m', 'u10', 'v10', 'ssrd']
V2_VARS = ['ssr', 'sp', 'stl1', 'sshf', 'swvl1', 'lmlt']


def convert_month(year, month):
    path1 = 'data/' + year + '/' + month + '/V1'
    path2 = 'data/' + year + '/' + month + '/V2'
    filename1 = os.listdir(path1)[0]
    filename2 = os.listdir(path2)[0]
    ds1 = xr.open_dataset(path1 + '/' + filename1)
    ds2 = xr.open_dataset(path2 + '/' + filename2)

    v1_data, v2_data = [], []
    for var in V1_VARS:
        data = ds1[var].values
        v1_data.append(data)
    for var in V2_VARS:
        data = ds2[var].values
        v2_data.append(data)

    ds1.close()
    ds2.close()
    all_vars = v1_data + v2_data
    stacked = np.stack(all_vars, axis=0)
    result = np.transpose(stacked, (1, 0, 2, 3))
    tensor = torch.from_numpy(result).float()
    return tensor

if __name__ == '__main__':
    path = 'data/9dc82e713982376547995ebb5e11a720.nc'
    
    all_data = []
    for year in YEARS:
        for month in MONTHS:
            monthly_data = convert_month(year, month)
            all_data.append(monthly_data)
            print(f"  {year}-{month}: shape {monthly_data.shape}")

    final_data = torch.cat(all_data, dim = 0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    torch.save(final_data, output_path)

