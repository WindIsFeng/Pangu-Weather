import os
import numpy as np
import onnx
import onnxruntime as ort
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import netCDF4 as nc

# Use GPU or CPU
use_GPU = True

# 定义基础目录路径
base_directory = "/disk1/code/AI_weather_models/Pangu/"

# 为每个风暴事件设置预测时长（小时）
prediction_hours = 48

# 定义多个时间段和对应的basin/name - 使用你提供的新格式
storm_events = [
    {"Basin": "WP", "Name": "Ragasa", "datetime": datetime(year=2025, month=9, day=22, hour=19, minute=0)},
]


def process_time_period(date_time, date_time_final, base_dir, basin, name):
    # 基于基础目录构建结果文件夹路径
    final_result_dir = os.path.join(
        os.path.join(base_dir, "output"),
        basin,
        name,
        (date_time.strftime("%Y-%m-%d-%H-%M") + "to" + date_time_final.strftime("%Y-%m-%d-%H-%M"))
    )
    os.makedirs(final_result_dir, exist_ok=True)

    # 基于基础目录构建临时文件夹路径
    temp_dir = os.path.join(base_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # 模型路径可以根据需要修改为绝对路径或相对于基础目录的路径
    model_6 = '/disk1/code/AI_weather_models/Pangu/model_file/pangu_weather_6.onnx'

    # 基于基础目录构建预测数据文件夹路径
    forecast_dir = os.path.join(
        os.path.join(base_dir, "data"),
        basin,
        name,
        date_time.strftime("%Y-%m-%d-%H-%M")
    )

    # 检查输入文件是否存在
    if not os.path.exists(os.path.join(forecast_dir, 'input_upper.npy')) or not os.path.exists(
            os.path.join(forecast_dir, 'input_surface.npy')):
        print(f"Input files not found in {forecast_dir}, skipping...")
        return False

    time_difference_in_hour = (date_time_final - date_time).total_seconds() / 3600
    current_date_time = date_time + timedelta(hours=6)
    last_date_time = date_time
    start = True
    ort_session = None

    while time_difference_in_hour >= 6:
        print(f"Remaining hours: {time_difference_in_hour}")

        model_used = model_6
        time_difference_in_hour -= 6
        print("Using 6-hour model")
        print("Current time:", current_date_time.strftime("%Y-%m-%d-%H-%M"))

        # 只在第一次迭代时加载模型
        if ort_session is None:
            model = onnx.load(model_used)

            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.enable_mem_reuse = False
            options.intra_op_num_threads = 30

            cuda_provider_options = {'arena_extend_strategy': 'kSameAsRequested'}

            if use_GPU:
                cuda_provider_options = {'device_id': 0}
                ort_session = ort.InferenceSession(model_used, sess_options=options,
                                                   providers=[('CUDAExecutionProvider', cuda_provider_options)])
            else:
                ort_session = ort.InferenceSession(model_used, sess_options=options, providers=['CPUExecutionProvider'])

        print("start prediction")

        input = None
        input_surface = None
        if start:
            input = np.load(os.path.join(forecast_dir, 'input_upper.npy')).astype(np.float32)
            input_surface = np.load(os.path.join(forecast_dir, 'input_surface.npy')).astype(np.float32)
            start = False
        else:
            input = np.load(os.path.join(final_result_dir,
                                         'output_upper_' + last_date_time.strftime("%Y-%m-%d-%H-%M") + '.npy')).astype(
                np.float32)
            input_surface = np.load(os.path.join(final_result_dir, 'output_surface_' + last_date_time.strftime(
                "%Y-%m-%d-%H-%M") + '.npy')).astype(np.float32)

        output, output_surface = ort_session.run(None, {'input': input, 'input_surface': input_surface})

        np.save(os.path.join(final_result_dir, 'output_upper_' + current_date_time.strftime("%Y-%m-%d-%H-%M")), output)
        np.save(os.path.join(final_result_dir, 'output_surface_' + current_date_time.strftime("%Y-%m-%d-%H-%M")),
                output_surface)

        last_date_time = current_date_time
        current_date_time += timedelta(hours=6)

    return True


def calculate_hours_since_epoch(dt):
    epoch = datetime(1970, 1, 1)
    return (dt - epoch).total_seconds() / 3600


def process_surface_files(results_dir, outputs_dir, date_time):
    surface_vars = {
        'msl': [],
        'u10': [],
        'v10': [],
        't2m': []
    }

    times = []
    current_time = date_time + timedelta(hours=6)
    time_increment = relativedelta(hours=6)

    for file in sorted(os.listdir(results_dir)):
        if file.endswith(".npy") and file.startswith("output_surface"):
            try:
                surface_data = np.load(os.path.join(results_dir, file))
                surface_vars['msl'].append(surface_data[0])
                surface_vars['u10'].append(surface_data[1])
                surface_vars['v10'].append(surface_data[2])
                surface_vars['t2m'].append(surface_data[3])
                times.append(calculate_hours_since_epoch(current_time))
                current_time += time_increment
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not surface_vars['u10']:
        print("No surface files found")
        return

    output_file = os.path.join(outputs_dir, "surface_combined.nc")
    with nc.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_file:
        nc_file.createDimension("valid_time", len(times))
        nc_file.createDimension("latitude", 721)
        nc_file.createDimension("longitude", 1440)

        nc_time = nc_file.createVariable("valid_time", np.float32, ("valid_time",))
        nc_lon = nc_file.createVariable("longitude", np.float32, ("longitude",))
        nc_lat = nc_file.createVariable("latitude", np.float32, ("latitude",))
        nc_msl = nc_file.createVariable("msl", np.float32, ("valid_time", "latitude", "longitude"))
        nc_u10 = nc_file.createVariable("u10", np.float32, ("valid_time", "latitude", "longitude"))
        nc_v10 = nc_file.createVariable("v10", np.float32, ("valid_time", "latitude", "longitude"))
        nc_t2m = nc_file.createVariable("t2m", np.float32, ("valid_time", "latitude", "longitude"))

        nc_time.units = "hours since 1970-01-01 00:00:00"
        nc_time.calendar = "standard"
        nc_lon.units = "degrees_east"
        nc_lat.units = "degrees_north"
        nc_u10.units = "m/s"
        nc_v10.units = "m/s"
        nc_t2m.units = "K"
        nc_msl.units = "Pa"

        nc_time[:] = np.array(times, dtype=np.float32)
        nc_lon[:] = np.linspace(0, 359.75, 1440)
        nc_lat[:] = np.linspace(90, -90, 721)

        for i in range(len(times)):
            nc_msl[i] = surface_vars['msl'][i]
            nc_u10[i] = surface_vars['u10'][i]
            nc_v10[i] = surface_vars['v10'][i]
            nc_t2m[i] = surface_vars['t2m'][i]


def process_upper_files(results_dir, outputs_dir, date_time):
    upper_vars = {
        'z': [],
        'q': [],
        't': [],
        'u': [],
        'v': []
    }

    times = []
    current_time = date_time + timedelta(hours=6)
    time_increment = relativedelta(hours=6)

    for file in sorted(os.listdir(results_dir)):
        if file.endswith(".npy") and file.startswith("output_upper"):
            try:
                upper_data = np.load(os.path.join(results_dir, file))
                upper_vars['z'].append(upper_data[0])
                upper_vars['q'].append(upper_data[1])
                upper_vars['t'].append(upper_data[2])
                upper_vars['u'].append(upper_data[3])
                upper_vars['v'].append(upper_data[4])
                times.append(calculate_hours_since_epoch(current_time))
                current_time += time_increment
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    if not upper_vars['z']:
        print("No upper files found")
        return

    output_file = os.path.join(outputs_dir, "upper_combined.nc")
    with nc.Dataset(output_file, "w", format="NETCDF4_CLASSIC") as nc_file:
        nc_file.createDimension("valid_time", len(times))
        nc_file.createDimension("longitude", 1440)
        nc_file.createDimension("latitude", 721)
        nc_file.createDimension("level", 13)

        nc_time = nc_file.createVariable("valid_time", np.float32, ("valid_time",))
        nc_lon = nc_file.createVariable("longitude", np.float32, ("longitude",))
        nc_lat = nc_file.createVariable("latitude", np.float32, ("latitude",))
        nc_level = nc_file.createVariable("level", np.int32, ("level",))
        nc_z = nc_file.createVariable("z", np.float32, ("valid_time", "level", "latitude", "longitude"))
        nc_q = nc_file.createVariable("q", np.float32, ("valid_time", "level", "latitude", "longitude"))
        nc_t = nc_file.createVariable("t", np.float32, ("valid_time", "level", "latitude", "longitude"))
        nc_u = nc_file.createVariable("u", np.float32, ("valid_time", "level", "latitude", "longitude"))
        nc_v = nc_file.createVariable("v", np.float32, ("valid_time", "level", "latitude", "longitude"))

        nc_time.units = "hours since 1970-01-01 00:00:00"
        nc_time.calendar = "standard"
        nc_lon.units = "degrees_east"
        nc_lat.units = "degrees_north"
        nc_level.units = "level"
        nc_level.long_name = "pressure_level"
        nc_z.units = "m**2 s**-2"
        nc_q.units = "kg/kg"
        nc_t.units = "K"
        nc_u.units = "m/s"
        nc_v.units = "m/s"

        nc_time[:] = np.array(times, dtype=np.float32)
        nc_level[:] = np.arange(1, 14)
        nc_lat[:] = np.linspace(90, -90, 721)
        nc_lon[:] = np.linspace(0, 359.75, 1440)

        for i in range(len(times)):
            nc_z[i] = upper_vars['z'][i]
            nc_q[i] = upper_vars['q'][i]
            nc_t[i] = upper_vars['t'][i]
            nc_u[i] = upper_vars['u'][i]
            nc_v[i] = upper_vars['v'][i]



# 模型推理阶段
for event in storm_events:
    start_time = event['datetime']
    end_time = start_time + timedelta(hours=prediction_hours)
    basin = event['Basin']
    name = event['Name']

    print(f"Processing {basin}/{name} from {start_time} to {end_time}")

    success = process_time_period(start_time, end_time, base_directory, basin, name)

    if success:
        # 合并输出为 netCDF
        results_dir = os.path.join(
            os.path.join(base_directory, "output"),
            basin,
            name,
            start_time.strftime("%Y-%m-%d-%H-%M") + "to" + end_time.strftime("%Y-%m-%d-%H-%M")
        )

        outputs_dir = os.path.join(
            os.path.join(base_directory, "output_combined"),
            basin,
            name,
            start_time.strftime("%Y-%m-%d-%H-%M") + "to" + end_time.strftime("%Y-%m-%d-%H-%M")
        )
        os.makedirs(outputs_dir, exist_ok=True)

        process_surface_files(results_dir, outputs_dir, start_time)
        process_upper_files(results_dir, outputs_dir, start_time)

        print(f"Processing complete. Output files created in: {outputs_dir}")
    else:
        print(f"Skipping {basin}/{name} due to missing input files")