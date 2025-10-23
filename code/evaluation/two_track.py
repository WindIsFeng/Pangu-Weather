import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import pandas as pd

# =================== 加载数据 ===================
df = xr.open_dataset(
    r"/ZXT/pangu/test/output/2023-07-23-06-00to2023-07-28-06-00/surface_combined.nc"
)
slp = df.msl.loc[:, :, :] / 100  # hPa

# 获取时间、经纬度
time_list = [pd.Timestamp(t).to_pydatetime() for t in slp.valid_time.values]
num_points = 20
selected_indices = np.linspace(0, len(time_list) - 1, num_points, dtype=int)
selected_times = [time_list[i] for i in selected_indices]

print(f"自动选取 {num_points} 个时间点:")
for t in selected_times:
    print(f"  - {t.strftime('%Y-%m-%d %H:%M')}")

lon_array, lat_array = [], []

# =================== 追踪最小 SLP ===================
for idx, date_judge in enumerate(selected_times):
    if idx == 0:
        slp_need = slp.sel(
            valid_time=date_judge,
            latitude=slice(35, 0),
            longitude=slice(90, 140)
        )
    else:
        prev_lat, prev_lon = lat_array[-1], lon_array[-1]
        lat_min, lat_max = max(prev_lat - 6, 0), min(prev_lat + 6, 35)
        lon_min, lon_max = max(prev_lon - 6, 90), min(prev_lon + 6, 140)
        slp_need = slp.sel(
            valid_time=date_judge,
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
        if slp_need.shape[0] == 0 or slp_need.shape[1] == 0:
            print(f"[警告] 第 {idx+1} 个时间点 ({date_judge}) 附近无数据，跳过该时刻")
            continue

    lat_grid, lon_grid = np.meshgrid(slp_need.latitude, slp_need.longitude, indexing='ij')
    min_index = np.unravel_index(np.nanargmin(slp_need.values), slp_need.shape)
    min_lat = float(lat_grid[min_index])
    min_lon = float(lon_grid[min_index])
    lat_array.append(min_lat)
    lon_array.append(min_lon)

    print(f"Time: {date_judge.strftime('%Y-%m-%d %H:%M')} | Min SLP: {np.nanmin(slp_need.values):.1f} hPa | "
          f"Location: ({min_lon:.2f}°E, {min_lat:.2f}°N)")

# =================== CMA 数据 ===================
cma_lon = [127.8, 127.1, 126.6, 126.3, 125.8, 125.1, 124.6, 123.7, 122.8,
           121.6, 121.3, 121.0, 120.6, 120.0, 119.7, 119.3, 119.1, 118.9, 118.8, 118.1]
cma_lat = [15.0, 15.1, 15.3, 15.7, 16.5, 16.9, 17.6, 18.3, 18.8,
           19.0, 18.9, 19.3, 19.6, 20.0, 20.7, 21.1, 21.8, 22.8, 24.1, 25.5]

# =================== 绘图 ===================
extent = [90, 140, 0, 35]
fig = plt.figure(figsize=[10, 8])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(extent, crs=ccrs.PlateCarree())

ax.coastlines('50m', linewidth=1.2)
ax.add_feature(cfeature.LAND, facecolor='#c14a09', alpha=0.8)
ax.add_feature(cfeature.OCEAN, facecolor='#add8e6', alpha=0.9)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)

gl = ax.gridlines(draw_labels=True, lw=1, color='gray', alpha=0.5, ls='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# Pangu轨迹（红色实线圆点）
ax.plot(lon_array, lat_array, 'ro-', markersize=4, label='Pangu Track')

# CMA轨迹（蓝色虚线方点）
ax.plot(cma_lon, cma_lat, 'bs--', markersize=4, label='CMA Track')

plt.title('Doksuri Typhoon Track: Pangu vs CMA', fontsize=14)
plt.legend(loc='lower left')

# 保存图像
# plt.savefig(
#     '/mnt/nfs/ssd/wangfei/wangfei_a100/ZXT/pangu/validation/Doksuri/pangu_vs_cma.png',
#     dpi=300, bbox_inches='tight'
# )
plt.show()
