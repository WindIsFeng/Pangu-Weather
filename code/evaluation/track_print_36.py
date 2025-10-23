import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import os
from datetime import datetime
import pandas as pd
import glob  # 用于批量查找文件

# 全局配置 - 请确认这些路径是否正确
ROOT_INPUT_DIR = "/mnt/nfs/hdd/wangfei/ZXT/model/Pangu/output_36"  # 输入根目录（output_36层级）
ROOT_OUTPUT_DIR = "/mnt/nfs/hdd/wangfei/ZXT/model/Evaluation/Pangu/36"  # 输出根目录
excel_path = "/mnt/nfs/hdd/wangfei/ZXT/model/Evaluation/combined_typhoons_converted.xlsx"  # 台风信息Excel
MERGED_EXCEL_PATH = os.path.join(ROOT_OUTPUT_DIR, "merged_all_results.xlsx")  # 合并后的总表路径
BASIN_CODES = ['SI', 'WP', 'EP', 'SP', 'NI', "Unknown"]  # 有效盆地代码
SEARCH_BUFFER_DEG = 5.0  # 未找到匹配时的搜索范围（度）
EXPECTED_TIME_PER_FILE = 6  # 【新增】每个NC文件预期的时间点数量（根据你的"文件数*6"需求设置）


def extract_path_info(nc_file_path):
    """从NC文件路径中提取BASIN、NAME、TIME信息"""
    path_parts = nc_file_path.split(os.sep)  # 按系统分隔符分割路径
    try:
        idx_output36 = path_parts.index("output_36")
        basin = path_parts[idx_output36 + 1]
        name = path_parts[idx_output36 + 2]
        time_str = path_parts[idx_output36 + 3]

        if basin not in BASIN_CODES:
            raise ValueError(f"无效的盆地代码: {basin}")
        return basin, name, time_str
    except (ValueError, IndexError) as e:
        raise ValueError(f"路径解析失败 {nc_file_path}: {str(e)}")


def plot_detail_map(lon, lat, data, wind_data, center_lon, center_lat,
                    time_str, pressure, wind_speed, output_dir, time_index):
    """绘制台风中心区域的详细图（等压线+中心标注）"""
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


def process_single_nc(nc_file_path):
    """
    处理单个NC文件的主函数
    返回：(处理结果元组, 处理详情字典)
    处理详情包含：文件路径、预期时间点、实际处理数、缺失时间点等
    """
    # 【新增】初始化处理详情变量
    process_detail = {
        "nc_file_path": nc_file_path,
        "basin": None,
        "name": None,
        "time_range": None,
        "expected_time_count": 0,  # NC文件实际包含的时间点数量
        "expected_time_per_file": EXPECTED_TIME_PER_FILE,  # 业务预期的时间点数量
        "actual_time_count": 0,  # 实际成功处理的时间点数量
        "missing_time_indices": [],  # 缺失的时间点索引（如[0,2]表示第1、3个时间点缺失）
        "status": "未处理",  # 状态：成功/部分成功/失败
        "error_msg": ""  # 错误信息（如有）
    }

    try:
        # 1. 提取路径信息并创建输出目录
        basin, name, time_str = extract_path_info(nc_file_path)
        process_detail["basin"] = basin
        process_detail["name"] = name
        process_detail["time_range"] = time_str
        print(f"\n===== 开始处理: {basin}/{name}/{time_str} =====")
        print(f"当前文件路径: {nc_file_path}")

        # 构建输出目录
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

        # 【新增】记录NC文件实际的时间点数量
        process_detail["expected_time_count"] = len(times)
        if process_detail["expected_time_count"] == 0:
            raise ValueError("NC文件中没有有效的时间数据")
        print(
            f"该NC文件实际包含 {process_detail['expected_time_count']} 个时间点（业务预期 {EXPECTED_TIME_PER_FILE} 个）")

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
        df_excel['time'] = pd.to_datetime(df_excel['time'], format='%m/%d/%Y %H:%M', errors='coerce')
        target_time = pd.to_datetime(first_time_str, format='%Y-%m-%d %H:%M', errors='coerce')

        # 精确匹配（BASIN/NAME/时间）
        match_row = df_excel[
            (df_excel['BASIN'] == basin) &
            (df_excel['NAME'] == name) &
            (df_excel['time'] == target_time)
            ]

        if not match_row.empty:
            start_lon = match_row.iloc[0]['min_lon']
            start_lat = match_row.iloc[0]['min_lat']
            print(f"从Excel获取起始点: ({start_lon:.2f}°E, {start_lat:.2f}°N)")
        else:
            name_matches = df_excel[df_excel['NAME'] == name]
            if name_matches.empty:
                print(f"Excel中未找到名称为 {name} 的任何记录，使用默认起始点(0,0)")
                start_lon, start_lat = 0.0, 0.0
            else:
                # 计算最接近时间点的位置
                name_matches['time_diff'] = (name_matches['time'] - target_time).abs()
                closest_row = name_matches.loc[name_matches['time_diff'].idxmin()]
                start_lon, start_lat = closest_row['longitude'], closest_row['latitude']
                print(f"Excel未找到精确匹配，使用最接近时间点的位置: ({start_lon:.2f}°E, {start_lat:.2f}°N)")

        # 4. 【核心修改】循环处理每个时间点（单个时间点失败不中断整个文件）
        track_results = []
        current_center_lon, current_center_lat = start_lon, start_lat

        for time_index in range(process_detail["expected_time_count"]):
            try:
                current_time = times[time_index]
                # 时间格式转换
                if isinstance(current_time, np.datetime64):
                    dt = current_time.astype('datetime64[s]').astype(datetime)
                    time_str_curr = dt.strftime("%Y-%m-%d %H:%M UTC")
                else:
                    time_str_curr = str(current_time)
                print(f"处理时间点 {time_index + 1}/{process_detail['expected_time_count']}: {time_str_curr}")

                # 获取当前时间层数据
                data = msl.isel(valid_time=time_index).values
                wind_data = wind_speed.isel(valid_time=time_index).values

                # 设置搜索范围
                buffer_deg = SEARCH_BUFFER_DEG if (
                            time_index == 0 and match_row.empty and not name_matches.empty) else 2.0
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

                # 定位台风中心（最低气压）
                min_idx = np.unravel_index(np.argmin(data_search), data_search.shape)
                min_lon = lon_search[min_idx[1]]
                min_lat = lat_search[min_idx[0]]
                min_pressure = data_search[min_idx]
                current_center_lon, current_center_lat = min_lon, min_lat  # 更新下一轮搜索中心

                # 计算最大风速（中心±2度）
                wind_mask_lon = (lon >= min_lon - 2) & (lon <= min_lon + 2)
                wind_mask_lat = (lat >= min_lat - 2) & (lat <= min_lat + 2)
                wind_search = wind_data[np.ix_(wind_mask_lat, wind_mask_lon)]
                max_wind_speed = wind_search.max() if wind_search.size > 0 else np.nan

                # 记录结果
                track_results.append({
                    'time': time_str_curr,
                    'min_lon': min_lon,
                    'min_lat': min_lat,
                    'min_pressure': min_pressure,
                    'max_wind_speed': max_wind_speed
                })
                print(
                    f"  ✅ 成功：中心({min_lon:.2f}, {min_lat:.2f})，气压{min_pressure:.2f}hPa，风速{max_wind_speed:.2f}m/s")

                # 绘制详情图
                plot_detail_map(
                    lon=lon, lat=lat, data=data, wind_data=wind_data,
                    center_lon=min_lon, center_lat=min_lat,
                    time_str=time_str_curr, pressure=min_pressure, wind_speed=max_wind_speed,
                    output_dir=output_dir, time_index=time_index
                )

                process_detail["actual_time_count"] += 1  # 成功计数+1

            except Exception as e:
                # 【新增】单个时间点处理失败，记录索引和错误
                missing_idx = time_index
                process_detail["missing_time_indices"].append(missing_idx)
                err_msg = f"时间点 {missing_idx + 1} 处理失败: {str(e)[:100]}"  # 截取前100字符避免过长
                process_detail["error_msg"] += f"; {err_msg}"
                print(f"  ❌ {err_msg}")
                continue  # 继续处理下一个时间点，不中断整个文件

        # 5. 保存结果（即使部分缺失，也保存已处理的记录）
        if track_results:
            df_results = pd.DataFrame(track_results)
            df_results.to_excel(output_excel_path, index=False)
            # 更新处理状态
            if process_detail["actual_time_count"] == process_detail["expected_time_count"]:
                process_detail["status"] = "成功（完整）"
            else:
                process_detail["status"] = "部分成功（有缺失）"
            print(f"\n===== 处理完成！已保存 {len(track_results)} 条记录至: {output_excel_path} =====")
            return (output_excel_path, basin, name, time_str), process_detail
        else:
            process_detail["status"] = "失败（无有效记录）"
            process_detail["error_msg"] += "; 所有时间点处理失败，无记录保存"
            print(f"\n===== 处理失败！该文件无有效记录保存 =====")
            return (None, None, None, None), process_detail

    except Exception as e:
        # 【新增】整个文件初始化失败（如路径解析、NC读取错误）
        process_detail["status"] = "失败（初始化错误）"
        process_detail["error_msg"] = f"文件初始化失败: {str(e)}"
        print(f"\n❌ 整个文件处理失败 {nc_file_path}: {str(e)}")
        return (None, None, None, None), process_detail


def merge_excel_files(process_results, total_actual_processed):
    """合并结果，并校验合并前后的记录数一致性"""
    print("\n===== 开始合并所有Excel结果 =====")
    all_data = []
    merged_count_before = 0  # 合并前统计的总记录数

    for result in process_results:
        excel_path, basin, name, time_str = result
        if not excel_path or not os.path.exists(excel_path):
            print(f"跳过无效文件: {excel_path}")
            continue

        try:
            df = pd.read_excel(excel_path)
            merged_count_before += len(df)
            # 添加元信息列
            df['BASIN'] = basin
            df['NAME'] = name
            df['TIME_RANGE'] = time_str
            # 调整列顺序
            cols = ['BASIN', 'NAME', 'TIME_RANGE', 'time', 'min_lon', 'min_lat', 'min_pressure', 'max_wind_speed']
            df = df[cols]
            all_data.append(df)
            print(f"已合并: {basin}/{name}/{time_str}，记录数: {len(df)}")
        except Exception as e:
            print(f"合并失败 {excel_path}: {str(e)}")

    if not all_data:
        print("没有可合并的数据")
        return

    # 合并数据
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_excel(MERGED_EXCEL_PATH, index=False)

    # 【新增】合并阶段记录数校验
    merged_count_after = len(merged_df)
    print(f"\n===== 合并完成！总表保存至: {MERGED_EXCEL_PATH} =====")
    print(f"1. 处理阶段统计的总记录数: {total_actual_processed}")
    print(f"2. 合并前统计的总记录数: {merged_count_before}")
    print(f"3. 合并后最终总记录数: {merged_count_after}")

    # 检查缺失
    if merged_count_before != merged_count_after:
        missing_merge = merged_count_before - merged_count_after
        print(f"⚠️  警告：合并过程中丢失 {missing_merge} 条记录！")
    if total_actual_processed != merged_count_before:
        missing_transfer = total_actual_processed - merged_count_before
        print(f"⚠️  警告：处理→合并阶段丢失 {missing_transfer} 条记录！")


def main():
    """主函数：批量处理+缺失数据统计"""
    print("===== 开始批量处理台风数据 =====")
    print(f"搜索目录: {ROOT_INPUT_DIR}")
    print(f"每个文件预期时间点数量: {EXPECTED_TIME_PER_FILE}")

    # 1. 查找所有NC文件
    pattern = os.path.join(ROOT_INPUT_DIR, '*', '*', '*', 'surface_combined.nc')
    nc_files = glob.glob(pattern)
    total_files = len(nc_files)
    print(f"找到 {total_files} 个NC文件待处理")
    if total_files == 0:
        print("警告：未找到任何NC文件，请检查路径是否正确")
        return

    # 2. 逐个处理文件
    process_results = []  # 用于合并的有效结果
    all_process_details = []  # 所有文件的处理详情（用于统计缺失）
    total_actual_processed = 0  # 所有文件实际处理的总记录数

    for file_idx, nc_file in enumerate(nc_files, 1):
        print(f"\n----- 处理文件 {file_idx}/{total_files} -----")
        result, detail = process_single_nc(nc_file)
        all_process_details.append(detail)
        if result[0]:  # 有有效输出才加入合并列表
            process_results.append(result)
        total_actual_processed += detail["actual_time_count"]  # 累加总记录数

    # 3. 【核心新增】打印缺失数据统计汇总
    print("\n" + "=" * 120)
    print("===== 缺失数据统计汇总 =====")
    print(f"总NC文件数: {total_files}")

    # 分类统计
    success_full = 0  # 完全成功（无缺失）
    success_partial = 0  # 部分成功（有缺失）
    failed_files = 0  # 完全失败
    total_expected_all = total_files * EXPECTED_TIME_PER_FILE  # 业务预期总记录数（文件数*6）
    total_actual_all = total_actual_processed  # 实际总记录数
    total_missing_all = total_expected_all - total_actual_all  # 总缺失记录数

    # 打印详细列表
    print("\n【文件处理详情列表】")
    print(
        f"{'文件索引':<6} {'BASIN':<6} {'NAME':<10} {'TIME_RANGE':<15} {'预期时间点':<12} {'实际处理':<10} {'缺失数':<8} {'状态':<15} {'缺失时间点索引':<20}")
    print("-" * 120)
    for idx, detail in enumerate(all_process_details, 1):
        basin = detail["basin"] or "-"
        name = detail["name"] or "-"
        time_range = detail["time_range"] or "-"
        exp_cnt = detail["expected_time_count"]
        act_cnt = detail["actual_time_count"]
        missing_cnt = exp_cnt - act_cnt
        status = detail["status"]
        missing_idx = ",".join(map(str, [x + 1 for x in detail["missing_time_indices"]])) if detail[
            "missing_time_indices"] else "无"
        print(
            f"{idx:<6} {basin:<6} {name:<10} {time_range:<15} {exp_cnt:<12} {act_cnt:<10} {missing_cnt:<8} {status:<15} {missing_idx:<20}")

        # 分类计数
        if status == "成功（完整）":
            success_full += 1
        elif status == "部分成功（有缺失）":
            success_partial += 1
        else:
            failed_files += 1

    # 打印总体统计
    print("\n【总体缺失统计】")
    print(f"1. 文件状态分布:")
    print(f"   - 完全成功（无缺失）: {success_full} 个")
    print(f"   - 部分成功（有缺失）: {success_partial} 个")
    print(f"   - 完全失败: {failed_files} 个")
    print(f"2. 记录数统计:")
    print(f"   - 业务预期总记录数（文件数×{EXPECTED_TIME_PER_FILE}）: {total_expected_all}")
    print(f"   - 实际处理总记录数: {total_actual_all}")
    print(f"   - 总缺失记录数: {total_missing_all}")
    print(
        f"   - 缺失率: {total_missing_all / total_expected_all * 100:.2f}%" if total_expected_all > 0 else "无预期数据")

    # 4. 合并结果
    merge_excel_files(process_results, total_actual_processed)

    print("\n===== 所有操作完成 =====")


if __name__ == "__main__":
    main()