import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Time points
time_points = [
'2023-07-22 00:00',
'2023-07-22 06:00',
'2023-07-22 12:00',
'2023-07-22 18:00',
'2023-07-23 00:00',
'2023-07-23 06:00',

]

# AI path data (Pangu轨迹)
ai_lon = [
261.25,
262.00,
262.75,
259.25,
259.25,
264.00,

]
ai_lat = [
14.25,
14.75,
15.00,
18.50,
18.50,
18.25,

]
# CMA data
# CMA数据（经纬度）- 第三列为纬度，第四列为经度
cma_lon = [261.7, 262.3, 262.8, 263.4, 263.9, 264.5]
cma_lat = [14.4, 14.8, 15.2, 15.7, 16.2, 16.7]
# Error calculation
lon_diff = np.array(ai_lon) - np.array(cma_lon)
lat_diff = np.array(ai_lat) - np.array(cma_lat)
eu_dist = np.sqrt(lon_diff**2 + lat_diff**2)
eu_km = eu_dist * 111

# Convert time format
time_dt = [datetime.strptime(x, "%Y-%m-%d %H:%M") for x in time_points]

# ======= Custom Y-Axis =======
# Set your y-axis start, end, and step (interval) here:
y_start = 0      # Minimum y-axis value
y_end = 500     # Maximum y-axis value
y_step = 50     # Tick interval

# ========== Plot ==========
plt.figure(figsize=(13,7))
plt.plot(time_dt, eu_km, marker='o', label='Distance Error (km)', linewidth=2)
# plt.plot(time_dt, lon_diff*111, marker='x', linestyle='', label='Longitude Error (km)')
# plt.plot(time_dt, lat_diff*111, marker='s', linestyle='', label='Latitude Error (km)')

plt.title('Typhoon Track Errors', fontsize=16)
plt.xlabel('Time', fontsize=13)
plt.ylabel('Distance Error (km)', fontsize=13)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.xticks(time_dt, [dt.strftime("%m-%d %H:%M") for dt in time_dt], rotation=45)

# ======= Set custom Y-Axis =======
plt.ylim(y_start, y_end)
plt.yticks(np.arange(y_start, y_end + y_step, y_step))
output_path = "/mnt/nfs/ssd/wangfei/wangfei_a100/model/Pangu/evaluation/doksuri/typhoon_track_error.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()
