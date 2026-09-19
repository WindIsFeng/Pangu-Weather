import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import os
from datetime import datetime

# 原始区域（大范围）
LON_MIN = 255
LON_MAX = 270
LAT_MIN = 10
LAT_MAX = 20

# 文件路径
file_path = "/mnt/nfs/hdd/wangfei/ZXT/model/Pangu/output_36/EP/AGATHA/2022-05-29-21-00to2022-05-31-09-00/surface_combined.nc"
output_dir = "/mnt/nfs/ssd/wangfei/wangfei_a100/ZXT/pangu/validation/Doksuri/test/"  # 修改为输出目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取 NetCDF 文件
ds = xr.open_dataset(file_path)
msl = ds['msl'] / 100  # Pa → hPa

# 获取风速数据（根据实际变量名调整）
# 情况1：如果有直接的风速变量
if 'wind_speed' in ds:
    wind_speed = ds['wind_speed']
# 情况2：如果只有u和v分量，需要计算风速
elif 'u10' in ds and 'v10' in ds:  # 10米风速分量
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']** 2)
else:
    # 根据实际数据中的风速变量名修改
    raise ValueError("未找到风速数据，请检查变量名")

lon = ds['longitude'].values
lat = ds['latitude'].values
times = ds['valid_time'].values  # 获取所有时间点

# 循环处理每个时间点
for time_index in range(len(times)):
    # 获取当前时间点
    current_time = times[time_index]

    # 转换为可读的时间格式
    if isinstance(current_time, np.datetime64):
        dt = current_time.astype('datetime64[s]').astype(datetime)
        time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    else:
        time_str = str(current_time)

    print(f"处理时间点: {time_str} (索引 {time_index}/{len(times) - 1})")

    # 获取当前时间点的气压和风速数据
    data = msl.isel(valid_time=time_index).values
    wind_data = wind_speed.isel(valid_time=time_index).values

    # 筛选初始大范围内的数据
    lon_mask_big = (lon >= LON_MIN) & (lon <= LON_MAX)
    lat_mask_big = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon_big = lon[lon_mask_big]
    lat_big = lat[lat_mask_big]
    data_big = data[np.ix_(lat_mask_big, lon_mask_big)]
    wind_big = wind_data[np.ix_(lat_mask_big, lon_mask_big)]

    # 找出最低气压及其索引位置
    min_index = np.unravel_index(np.argmin(data_big), data_big.shape)
    min_lat = lat_big[min_index[0]]
    min_lon = lon_big[min_index[1]]
    min_pressure = data_big[min_index]

    # 找出最大风速及其索引位置
    max_wind_index = np.unravel_index(np.argmax(wind_big), wind_big.shape)
    max_wind_lat = lat_big[max_wind_index[0]]
    max_wind_lon = lon_big[max_wind_index[1]]
    max_wind_speed = wind_big[max_wind_index]

    print(f"({min_lon:.2f}°E, {min_lat:.2f}°N), 气压: {min_pressure:.2f} hPa")

    print(f"  最大风速点:风速: {max_wind_speed:.2f} m/s")

    # 设置 ±1° 范围的小区域，以最低点为中心
    buffer_deg = 20.0
    LON_MIN_FOCUS = min_lon - buffer_deg
    LON_MAX_FOCUS = min_lon + buffer_deg
    LAT_MIN_FOCUS = min_lat - buffer_deg
    LAT_MAX_FOCUS = min_lat + buffer_deg

    # 保证不超出数据边界
    LON_MIN_FOCUS = max(LON_MIN_FOCUS, lon.min())
    LON_MAX_FOCUS = min(LON_MAX_FOCUS, lon.max())
    LAT_MIN_FOCUS = max(LAT_MIN_FOCUS, lat.min())
    LAT_MAX_FOCUS = min(LAT_MAX_FOCUS, lat.max())

    # 重新筛选小区域数据
    lon_mask = (lon >= LON_MIN_FOCUS) & (lon <= LON_MAX_FOCUS)
    lat_mask = (lat >= LAT_MIN_FOCUS) & (lat <= LAT_MAX_FOCUS)
    lon_focus = lon[lon_mask]
    lat_focus = lat[lat_mask]
    data_focus = data[np.ix_(lat_mask, lon_mask)]
    wind_focus = wind_data[np.ix_(lat_mask, lon_mask)]

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    m = Basemap(
        projection='cyl',
        llcrnrlat=LAT_MIN_FOCUS, urcrnrlat=LAT_MAX_FOCUS,
        llcrnrlon=LON_MIN_FOCUS, urcrnrlon=LON_MAX_FOCUS,
        resolution='c',
        ax=ax
    )

    lon_grid, lat_grid = np.meshgrid(lon_focus, lat_focus)
    x, y = m(lon_grid, lat_grid)
    contour = m.contourf(x, y, data_focus, levels=60, cmap='coolwarm', latlon=False)

    # 地图特征
    m.drawcoastlines(linewidth=0.8)
    m.drawcountries(linewidth=0.5, linestyle=':')

    # 网格线
    meridians = np.arange(np.floor(LON_MIN_FOCUS), np.ceil(LON_MAX_FOCUS) + 1, 0.5)
    parallels = np.arange(np.floor(LAT_MIN_FOCUS), np.ceil(LAT_MAX_FOCUS) + 1, 0.5)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.5, color='gray', dashes=[1, 1])
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.5, color='gray', dashes=[1, 1])

    # 标注最低点和最大风速点
    plt.plot(min_lon, min_lat, marker='*', color='black', markersize=15,
             label=f'{min_pressure:.2f} hPa')
    plt.plot(max_wind_lon, max_wind_lat, marker='D', color='yellow', markersize=10,
             label=f'{max_wind_speed:.2f} m/s')
    plt.legend(loc='upper right')

    # 添加标题
    plt.title(f"{time_str}\n"
              f" ({min_lon:.2f}°E, {min_lat:.2f}°N)\n",
              fontsize=14)

    # 添加色标
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('MSLP (hPa)', fontsize=12)

    # 生成输出路径
    filename = f"pangu_focus_min_pressure_{time_index:03d}.png"
    output_path = os.path.join(output_dir, filename)

    # 保存图像（取消注释即可保存）
    # plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)  # 关闭图形以释放内存


print(f"所有时间点处理完成！图像已保存至: {output_dir}")
