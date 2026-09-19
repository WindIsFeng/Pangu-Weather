import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import os
from datetime import datetime
import pandas as pd
import glob  # 用于批量查找文件

# 全局配置 - 请确认这些路径是否正确
ROOT_INPUT_DIR = "/mnt/nfs/hdd/wangfei/ZXT/model/Pangu/output_60"  # 输入根目录（output_36层级）
ROOT_OUTPUT_DIR = "/mnt/nfs/hdd/wangfei/ZXT/model/Evaluation/Pangu/60"  # 输出根目录
excel_path = "/mnt/nfs/hdd/wangfei/ZXT/model/Evaluation/combined_typhoons_converted.xlsx"  # 台风信息Excel
MERGED_EXCEL_PATH = os.path.join(ROOT_OUTPUT_DIR, "merged_all_results.xlsx")  # 合并后的总表路径
BASIN_CODES = ['SI', 'WP', 'EP', 'SP', 'NI', "Unknown"]  # 有效盆地代码
SEARCH_BUFFER_DEG = 5.0  # 未找到匹配时的搜索范围（度）

def extract_path_info(nc_file_path):
    """从NC文件路径中提取BASIN、NAME、TIME信息"""
    path_parts = nc_file_path.split(os.sep)  # 按系统分隔符分割路径
    try:
        # 定位output_36的索引，后续依次为BASIN/NAME/TIME
        idx_output36 = path_parts.index("output_60")
        basin = path_parts[idx_output36 + 1]
        name = path_parts[idx_output36 + 2]
        time_str = path_parts[idx_output36 + 3]

        # 验证盆地代码有效性
        if basin not in BASIN_CODES:
            raise ValueError(f"无效的盆地代码: {basin}")
        return basin, name, time_str
    except (ValueError, IndexError) as e:
        raise ValueError(f"路径解析失败 {nc_file_path}: {str(e)}")


def process_single_nc(nc_file_path):
    """处理单个NC文件的主函数，返回处理结果路径和元信息"""
    try:
        # 1. 提取路径信息并创建输出目录
        basin, name, time_str = extract_path_info(nc_file_path)
        print(f"\n===== 开始处理: {basin}/{name}/{time_str} =====")

        # 构建输出目录（保持BASIN/NAME/TIME层级）
        output_dir = os.path.join(ROOT_OUTPUT_DIR, basin, name, time_str)
        os.makedirs(output_dir, exist_ok=True)
        output_excel_path = os.path.join(output_dir, "track_results.xlsx")

        # 2. 读取NC文件数据
        ds = xr.open_dataset(nc_file_path)
        msl = ds['msl'] / 100  # Pa转hPa

        # 获取风速数据（兼容不同变量名）
        if 'wind_speed' in ds:
            wind_speed = ds['wind_speed']
        elif 'u10' in ds and 'v10' in ds:
            wind_speed = np.sqrt(ds['u10'] ** 2 + ds['v10'] ** 2)
        else:
            raise ValueError("未找到风速数据（需wind_speed或u10/v10变量）")

        lon = ds['longitude'].values
        lat = ds['latitude'].values
        times = ds['valid_time'].values

        if len(times) == 0:
            raise ValueError("NC文件中没有有效的时间数据")

        # 3. 从Excel获取起始点（匹配BASIN/NAME/时间）
        first_time = times[0]
        if isinstance(first_time, np.datetime64):
            first_time_dt = first_time.astype('datetime64[s]').astype(datetime)
            first_time_str = first_time_dt.strftime("%Y-%m-%d %H:%M")
        else:
            first_time_str = str(first_time)
        print(f"NC文件首个时间点: {first_time_str}")

        # 读取Excel并匹配起始点
        df_excel = pd.read_excel(excel_path)
        df_excel['time'] = pd.to_datetime(df_excel['time'], format='%m/%d/%Y %H:%M', errors='coerce')  # 统一时间格式
        target_time = pd.to_datetime(first_time_str, format='%Y-%m-%d %H:%M', errors='coerce')

        # 精确匹配（BASIN/NAME/时间）
        match_row = df_excel[
            (df_excel['BASIN'] == basin) &
            (df_excel['NAME'] == name) &
            (df_excel['time'] == target_time)
            ]

        if not match_row.empty:
            # 找到精确匹配的起始点
            start_lon = match_row.iloc[0]['min_lon']
            start_lat = match_row.iloc[0]['min_lat']
            print(f"从Excel获取起始点: ({start_lon:.2f}°E, {start_lat:.2f}°N)")
        else:
            # 未找到精确匹配，尝试在相同NAME下查找所有记录
            name_matches = df_excel[df_excel['NAME'] == name]

            if name_matches.empty:
                # 该名称在Excel中无任何记录，使用默认值
                print(f"Excel中未找到名称为 {name} 的任何记录，使用默认起始点")
                start_lon, start_lat = 0.0, 0.0  # 默认起始点
            else:
                # 计算该名称下所有经纬度的平均值作为起始点
                avg_lon = name_matches['longitude'].mean()
                avg_lat = name_matches['latitude'].mean()
                start_lon, start_lat = avg_lon, avg_lat

                # 计算时间差，找到最接近的时间点
                name_matches['time_diff'] = (name_matches['time'] - target_time).abs()
                closest_row = name_matches.loc[name_matches['time_diff'].idxmin()]
                closest_lon = closest_row['longitude']
                closest_lat = closest_row['latitude']

                print(f"Excel中未找到精确匹配的起始点，使用名称为 {name} 的记录计算起始点")
                print(f"  所有记录的平均位置: ({avg_lon:.2f}°E, {avg_lat:.2f}°N)")
                print(f"  最接近时间点的位置: ({closest_lon:.2f}°E, {closest_lat:.2f}°N)")
                print(f"  将在该位置周围{SEARCH_BUFFER_DEG}°范围内搜索")

                # 使用最接近时间点的位置作为起始点
                start_lon, start_lat = closest_lon, closest_lat

        # 4. 循环处理每个时间点的台风追踪
        track_results = []
        current_center_lon, current_center_lat = start_lon, start_lat  # 初始搜索中心

        for time_index in range(len(times)):
            # 时间格式转换
            current_time = times[time_index]
            if isinstance(current_time, np.datetime64):
                dt = current_time.astype('datetime64[s]').astype(datetime)
                time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
            else:
                time_str = str(current_time)
            print(f"处理时间点 {time_index + 1}/{len(times)}: {time_str}")

            # 获取当前时间层数据
            data = msl.isel(valid_time=time_index).values
            wind_data = wind_speed.isel(valid_time=time_index).values

            # 设置搜索范围（首时间点使用5°范围，后续2度，避免漂移）
            if time_index == 0 and match_row.empty and not name_matches.empty:
                buffer_deg = SEARCH_BUFFER_DEG  # 未找到匹配时使用更大的初始搜索范围
            else:
                buffer_deg = 1.0 if time_index == 0 else 2.0

            LON_MIN = max(current_center_lon - buffer_deg, lon.min())
            LON_MAX = min(current_center_lon + buffer_deg, lon.max())
            LAT_MIN = max(current_center_lat - buffer_deg, lat.min())
            LAT_MAX = min(current_center_lat + buffer_deg, lat.max())

            # 筛选搜索区域数据
            lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
            lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
            lon_search = lon[lon_mask]
            lat_search = lat[lat_mask]
            data_search = data[np.ix_(lat_mask, lon_mask)]

            # 定位最低气压中心（台风中心）
            min_idx = np.unravel_index(np.argmin(data_search), data_search.shape)
            min_lon = lon_search[min_idx[1]]
            min_lat = lat_search[min_idx[0]]
            min_pressure = data_search[min_idx]

            # 更新下一时间点搜索中心
            current_center_lon, current_center_lat = min_lon, min_lat

            # 搜索最大风速（中心±2度范围内）
            wind_mask_lon = (lon >= min_lon - 2) & (lon <= min_lon + 2)
            wind_mask_lat = (lat >= min_lat - 2) & (lat <= min_lat + 2)
            wind_search = wind_data[np.ix_(wind_mask_lat, wind_mask_lon)]
            max_wind_speed = wind_search.max() if wind_search.size > 0 else np.nan

            print(
                f"  台风中心: ({min_lon:.2f}°E, {min_lat:.2f}°N), 最低气压: {min_pressure:.2f} hPa, 最大风速: {max_wind_speed:.2f} m/s")

            # 保存结果
            track_results.append({
                'time': time_str,
                'min_lon': min_lon,
                'min_lat': min_lat,
                'min_pressure': min_pressure,
                'max_wind_speed': max_wind_speed
            })

            # 绘制并保存详细图（中心±1度范围）
            plot_detail_map(
                lon=lon, lat=lat, data=data, wind_data=wind_data,
                center_lon=min_lon, center_lat=min_lat,
                time_str=time_str, pressure=min_pressure, wind_speed=max_wind_speed,
                output_dir=output_dir, time_index=time_index
            )

        # 5. 保存追踪结果到Excel
        df_results = pd.DataFrame(track_results)
        df_results.to_excel(output_excel_path, index=False)
        print(f"\n===== 处理完成！结果保存至: {output_dir} =====")

        return output_excel_path, basin, name, time_str

    except Exception as e:
        print(f"\n处理失败 {nc_file_path}: {str(e)}")
        return None, None, None, None


def plot_detail_map(lon, lat, data, wind_data, center_lon, center_lat,
                    time_str, pressure, wind_speed, output_dir, time_index):
    """绘制台风中心区域的详细图（等压线+中心标注）"""
    # 设置绘图范围（中心±1度）
    detail_deg = 1.0
    LON_MIN = max(center_lon - detail_deg, lon.min())
    LON_MAX = min(center_lon + detail_deg, lon.max())
    LAT_MIN = max(center_lat - detail_deg, lat.min())
    LAT_MAX = min(center_lat + detail_deg, lat.max())

    # 筛选绘图区域数据
    lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
    lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    lon_detail = lon[lon_mask]
    lat_detail = lat[lat_mask]
    data_detail = data[np.ix_(lat_mask, lon_mask)]

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # 初始化地图
    m = Basemap(
        projection='cyl',
        llcrnrlat=LAT_MIN, urcrnrlat=LAT_MAX,
        llcrnrlon=LON_MIN, urcrnrlon=LON_MAX,
        resolution='i', ax=ax
    )
    lon_grid, lat_grid = np.meshgrid(lon_detail, lat_detail)
    x, y = m(lon_grid, lat_grid)

    # 绘制等压线（MSLP）
    contour = m.contourf(x, y, data_detail, levels=20, cmap='coolwarm', latlon=False)
    m.drawcoastlines(linewidth=0.8)
    m.drawcountries(linewidth=0.5, linestyle=':')

    # 添加网格线
    m.drawmeridians(np.arange(LON_MIN, LON_MAX + 0.1, 0.2),
                    labels=[0, 0, 0, 1], linewidth=0.3, color='gray')
    m.drawparallels(np.arange(LAT_MIN, LAT_MAX + 0.1, 0.2),
                    labels=[1, 0, 0, 0], linewidth=0.3, color='gray')

    # 标注台风中心和信息
    plt.plot(center_lon, center_lat, marker='*', color='black', markersize=15,
             label=f'{pressure:.2f} hPa')
    plt.legend(loc='upper right')
    plt.title(f"{time_str}\n({center_lon:.2f}°E, {center_lat:.2f}°N)\n{wind_speed:.2f} m/s",
              fontsize=12)

    # 添加色标
    cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('MSLP (hPa)', fontsize=10)

    # 保存图片
    img_path = os.path.join(output_dir, f"detail_{time_index:03d}.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def merge_excel_files(process_results):
    """合并所有处理结果Excel文件，添加BASIN、NAME和TIME信息"""
    print("\n===== 开始合并所有Excel结果 =====")

    all_data = []
    for result in process_results:
        excel_path, basin, name, time_str = result
        if not excel_path or not os.path.exists(excel_path):
            print(f"跳过无效文件: {excel_path}")
            continue

        try:
            # 读取单个Excel文件
            df = pd.read_excel(excel_path)

            # 添加元信息列
            df['BASIN'] = basin
            df['NAME'] = name
            df['TIME_RANGE'] = time_str  # 原始文件的时间范围

            # 调整列顺序，将元信息放在前面
            cols = ['BASIN', 'NAME', 'TIME_RANGE', 'time', 'min_lon', 'min_lat', 'min_pressure', 'max_wind_speed']
            df = df[cols]

            all_data.append(df)
            print(f"已合并: {basin}/{name}/{time_str}")
        except Exception as e:
            print(f"合并失败 {excel_path}: {str(e)}")

    if not all_data:
        print("没有可合并的数据")
        return

    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)

    # 保存合并结果
    merged_df.to_excel(MERGED_EXCEL_PATH, index=False)
    print(f"\n===== 合并完成！总表保存至: {MERGED_EXCEL_PATH} =====")
    print(f"总记录数: {len(merged_df)} 条")


def main():
    """主函数：查找所有NC文件并批量处理，最后合并结果"""
    print("===== 开始批量处理台风数据 =====")
    print(f"搜索目录: {ROOT_INPUT_DIR}")

    # 查找所有符合路径模式的NC文件
    # 路径模式：ROOT_INPUT_DIR/BASIN/NAME/TIME/surface_combined.nc
    pattern = os.path.join(ROOT_INPUT_DIR, '*', '*', '*', 'surface_combined.nc')
    nc_files = glob.glob(pattern)

    # 显示查找结果
    print(f"找到 {len(nc_files)} 个NC文件待处理")
    if not nc_files:
        print("警告：未找到任何NC文件，请检查路径是否正确")
        return

    # 逐个处理文件并记录结果路径
    process_results = []
    for i, nc_file in enumerate(nc_files, 1):
        print(f"\n----- 处理文件 {i}/{len(nc_files)} -----")
        result = process_single_nc(nc_file)
        if result[0]:  # 如果处理成功
            process_results.append(result)

    # 合并所有结果
    merge_excel_files(process_results)

    print("\n===== 所有操作完成 =====")


if __name__ == "__main__":
    main()  # 触发批量处理