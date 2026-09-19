import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import pandas as pd

# =================== 加载数据 ===================
df = xr.open_dataset(
    "/mnt/nfs/hdd/wangfei/ZXT/model/Pangu/output_36/WP/TALIM/2023-07-12-18-00to2023-07-14-06-00/surface_combined.nc"
)
slp = df.msl.loc[:, :, :] / 100  # 转换为hPa

# 获取时间、经纬度
time_list = [pd.Timestamp(t).to_pydatetime() for t in slp.valid_time.values]
num_points = 20
selected_indices = np.linspace(0, len(time_list) - 1, num_points, dtype=int)
selected_times = [time_list[i] for i in selected_indices]

print(f"自动选取 {num_points} 个时间点:")
for t in selected_times:
    print(f"  - {t.strftime('%Y-%m-%d %H:%M')}")

lon_array, lat_array = [], []
valid_times = []  # 存储有效时间点（剔除跳过的时刻）

# =================== 追踪最小 SLP ===================
for idx, date_judge in enumerate(selected_times):
    if idx == 0:
        # 初始时刻：在较大范围搜索
        slp_need = slp.sel(
            valid_time=date_judge,
            latitude=slice(35, 0),
            longitude=slice(90, 140)
        )
    else:
        # 后续时刻：基于前一时刻位置缩小搜索范围
        prev_lat, prev_lon = lat_array[-1], lon_array[-1]
        lat_min, lat_max = max(prev_lat - 6, 0), min(prev_lat + 6, 35)
        lon_min, lon_max = max(prev_lon - 6, 90), min(prev_lon + 6, 140)
        slp_need = slp.sel(
            valid_time=date_judge,
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
        # 检查是否有数据，无数据则跳过
        if slp_need.shape[0] == 0 or slp_need.shape[1] == 0:
            print(f"[警告] 第 {idx+1} 个时间点 ({date_judge}) 附近无数据，跳过该时刻")
            continue

    # 计算最小SLP位置
    lat_grid, lon_grid = np.meshgrid(slp_need.latitude, slp_need.longitude, indexing='ij')
    min_index = np.unravel_index(np.nanargmin(slp_need.values), slp_need.shape)
    min_lat = float(lat_grid[min_index])
    min_lon = float(lon_grid[min_index])
    lat_array.append(min_lat)
    lon_array.append(min_lon)
    valid_times.append(date_judge)  # 记录有效时间点

    print(f"Time: {date_judge.strftime('%Y-%m-%d %H:%M')} | Min SLP: {np.nanmin(slp_need.values):.1f} hPa | "
          f"Location: ({min_lon:.2f}°E, {min_lat:.2f}°N)")

# =================== 按要求格式输出数据 ===================
print("\n# 时间点数据")
print("time_points = [")
for t in valid_times:
    print(f"'{t.strftime('%Y-%m-%d %H:%M')}',")
print("]")

print("\n# AI path data (Pangu轨迹)")
print("ai_lon = [")
for lon in lon_array:
    print(f"{lon:.2f},")
print("]")

print("ai_lat = [")
for lat in lat_array:
    print(f"{lat:.2f},")
print("]")

# =================== 绘图（仅Pangu轨迹） ===================
extent = [90, 140, 0, 35]  # 地图范围：[经度_min, 经度_max, 纬度_min, 纬度_max]
fig = plt.figure(figsize=[10, 8])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(extent, crs=ccrs.PlateCarree())

# 添加地图特征
ax.coastlines('50m', linewidth=1.2)  # 海岸线
ax.add_feature(cfeature.LAND, facecolor='#c14a09', alpha=0.8)  # 陆地
ax.add_feature(cfeature.OCEAN, facecolor='#add8e6', alpha=0.9)  # 海洋
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)  # 国界

# 添加网格线
gl = ax.gridlines(draw_labels=True, lw=1, color='gray', alpha=0.5, ls='--')
gl.top_labels = False  # 关闭顶部标签
gl.right_labels = False  # 关闭右侧标签
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}

# 绘制Pangu轨迹
ax.plot(lon_array, lat_array, 'ro-', markersize=6, label='Pangu Track', linewidth=1.5)

# 添加标题和图例
plt.title('Doksuri Typhoon Track (Pangu Model)', fontsize=14, pad=15)
plt.legend(loc='lower left', fontsize=10)

# 保存图像（如需）
# plt.savefig(
#     '/mnt/nfs/ssd/wangfei/wangfei_a100/ZXT/pangu/validation/Doksuri/pangu_track.png',
#     dpi=300, bbox_inches='tight'
# )

plt.show()