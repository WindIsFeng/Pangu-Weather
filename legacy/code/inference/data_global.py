import cdsapi
import numpy as np
import netCDF4 as nc
import os
from datetime import datetime

c = cdsapi.Client()

# Example of the new input datetime format
date_times = [
    {"Basin": "WP", "Name": "Ragasa", "datetime": datetime(year=2025, month=9, day=22, hour=19, minute=0)},
]

# The variables required
surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

# Area to download
area = [90, 0, -90, 360]

# Pressure levels required
pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']

def download_and_process_data(basin, name, date_time):
    # Define the base directory
    base_dir = "/disk1/code/AI_weather_models/Pangu/"

    # Create the folder structure: BASIN -> name -> datetime
    ZXT_dir = os.path.join(
        os.path.join(base_dir, "data"),
        basin,
        name,
        date_time.strftime("%Y-%m-%d-%H-%M"),
    )
    os.makedirs(ZXT_dir, exist_ok=True)

    print(f"Processing data for {basin}_{name} at {date_time.strftime('%Y-%m-%d %H:%M')}...")

    try:
        # Download the surface data
        c.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': surface_variables,
            'date': date_time.strftime("%Y-%m-%d"),
            'time': date_time.strftime("%H:%M"),
            'area': area,
        }, os.path.join(ZXT_dir, 'surface.nc'))

        # Download the upper air data
        c.retrieve('reanalysis-era5-pressure-levels', {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': upper_variables,
            'pressure_level': pressure_levels,
            'date': date_time.strftime("%Y-%m-%d"),
            'time': date_time.strftime("%H:%M"),
            'area': area,
        }, os.path.join(ZXT_dir, 'upper.nc'))

        # Convert the surface data to npy
        surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
        with nc.Dataset(os.path.join(ZXT_dir, 'surface.nc')) as nc_file:
            surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
            surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
            surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
            surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
        np.save(os.path.join(ZXT_dir, 'input_surface.npy'), surface_data)

        # Convert the upper air data to npy
        upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
        with nc.Dataset(os.path.join(ZXT_dir, 'upper.nc')) as nc_file:
            upper_data[0] = (nc_file.variables['z'][:]).astype(np.float32)
            upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
            upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
            upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
            upper_data[4] = nc_file.variables['v'][:].astype(np.float32)
        np.save(os.path.join(ZXT_dir, 'input_upper.npy'), upper_data)

        print(f"Successfully processed data for {basin}_{name} at {date_time.strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        print(f"Error processing data for {basin}_{name} at {date_time.strftime('%Y-%m-%d %H:%M')}: {str(e)}")


# Process all date-times
for item in date_times:
    download_and_process_data(item['Basin'], item['Name'], item['datetime'])

print("All downloads and processing completed.")
