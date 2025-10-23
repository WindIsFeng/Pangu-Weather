import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import os
from datetime import datetime
import glob  # 用于批量获取文件路径


# -------------------------- 配置参数（请根据需求修改） --------------------------
# 原始区域（大范围）
LON_MIN = -100
LON_MAX = -92
LAT_MIN = 10
LAT_MAX = 20

# 输入文件夹（存放所有要处理的.nc文件）
input_dir = '/model/Era/mslp_data/EP/AGATHA'  # 修改为你的输入文件夹路径
# 输出目录（所有图像会保存在这里，不同文件的图像通过文件名区分）
output_dir = "/ZXT/pangu/validation/Doksuri/test/"
# ------------------------------------------------------------------------------


# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取输入文件夹内所有.nc文件（自动筛选后缀为.nc的文件）
nc_files = glob.glob(os.path.join(input_dir, "*.nc"))

# 检查是否有NC文件
if not nc_files:
    print(f"警告：在输入文件夹 {input_dir} 中未找到任何.nc文件，请检查路径是否正确！")
    exit()  # 无文件时退出程序


# 循环处理每个NC文件
for file_idx, nc_file in enumerate(nc_files):
    # 获取文件名（不含路径和后缀，用于区分输出图像）
    file_basename = os.path.splitext(os.path.basename(nc_file))[0]
    print(f"\n===== 开始处理文件 {file_idx+1}/{len(nc_files)}：{nc_file} =====")

    try:
        # 读取当前NC文件
        ds = xr.open_dataset(nc_file)
        msl = ds['msl'] / 100  # Pa → hPa
        lon = ds['longitude'].values
        lat = ds['latitude'].values
        times = ds['valid_time'].values  # 获取当前文件的所有时间点

        # 检查时间点是否存在
        if len(times) == 0:
            print(f"文件 {file_basename} 中无时间点数据，跳过该文件")
            continue

        # 循环处理当前文件的每个时间点
        for time_index in range(len(times)):
            # 获取当前时间点
            current_time = times[time_index]

            # 转换为可读的时间格式
            if isinstance(current_time, np.datetime64):
                dt = current_time.astype('datetime64[s]').astype(datetime)
                time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
            else:
                time_str = str(current_time)

            print(f"处理时间点: {time_str} (文件 {file_idx+1}/{len(nc_files)}，时间索引 {time_index}/{len(times)-1})")

            # 获取当前时间点的气压数据
            data = msl.isel(valid_time=time_index).values

            # 筛选初始大范围内的数据
            lon_mask_big = (lon >= LON_MIN) & (lon <= LON_MAX)
            lat_mask_big = (lat >= LAT_MIN) & (lat <= LAT_MAX)
            lon_big = lon[lon_mask_big]
            lat_big = lat[lat_mask_big]
            data_big = data[np.ix_(lat_mask_big, lon_mask_big)]

            # 检查筛选后的数据是否为空（避免报错）
            if data_big.size == 0:
                print(f"  警告：当前时间点在区域 {LON_MIN}-{LON_MAX}°E, {LAT_MIN}-{LAT_MAX}°N 内无数据，跳过该时间点")
                continue

            # 找出最小值及其索引位置
            min_index = np.unravel_index(np.argmin(data_big), data_big.shape)
            min_lat = lat_big[min_index[0]]
            min_lon = lon_big[min_index[1]]
            min_pressure = data_big[min_index]
            print(f"  最低气压点: ({min_lon:.2f}°E, {min_lat:.2f}°N), 气压: {min_pressure:.2f} hPa")

            # 设置±1°范围的小区域（以最低点为中心）
            buffer_deg = 1.0
            LON_MIN_FOCUS = min_lon - buffer_deg
            LON_MAX_FOCUS = min_lon + buffer_deg
            LAT_MIN_FOCUS = min_lat - buffer_deg
            LAT_MAX_FOCUS = min_lat + buffer_deg

            # 保证不超出数据边界
            LON_MIN_FOCUS = max(LON_MIN_FOCUS, lon.min())
            LON_MAX_FOCUS = min(LON_MIN_FOCUS, lon.max())  # 这里原代码可能笔误，修正为min(LON_MAX_FOCUS, lon.max())
            LAT_MIN_FOCUS = max(LAT_MIN_FOCUS, lat.min())
            LAT_MAX_FOCUS = min(LAT_MAX_FOCUS, lat.max())

            # 重新筛选小区域数据
            lon_mask = (lon >= LON_MIN_FOCUS) & (lon <= LON_MAX_FOCUS)
            lat_mask = (lat >= LAT_MIN_FOCUS) & (lat <= LAT_MAX_FOCUS)
            lon_focus = lon[lon_mask]
            lat_focus = lat[lat_mask]
            data_focus = data[np.ix_(lat_mask, lon_mask)]

            # 检查小区域数据是否为空
            if data_focus.size == 0:
                print(f"  警告：小区域内无数据，跳过绘图")
                continue

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

            # 标注最低点
            plt.plot(min_lon, min_lat, marker='*', color='black', markersize=15, label=f'{min_pressure:.2f} hPa')
            plt.legend(loc='upper right')

            # 添加标题（包含文件名，方便区分）
            plt.title(f"{file_basename}\n{time_str}\n({min_lon:.2f}°E, {min_lat:.2f}°N)", fontsize=14)

            # 添加色标
            cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
            cbar.set_label('MSLP (hPa)', fontsize=12)

            # 生成输出路径（文件名包含原文件标识+时间索引，避免覆盖）
            filename = f"{file_basename}_time_{time_index:03d}_pangu.png"
            output_path = os.path.join(output_dir, filename)

            # 保存图像（如果需要显示图像，保留plt.show()；若批量处理，建议注释掉以提高速度）
            plt.show()
            # plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)  # 关闭图形释放内存

        print(f"文件 {file_basename} 处理完成！\n")

    except Exception as e:
        # 捕获文件处理中的错误，避免程序中断
        print(f"处理文件 {file_basename} 时出错：{str(e)}，跳过该文件")
        continue


print(f"所有文件处理完成！图像已保存至: {output_dir}")