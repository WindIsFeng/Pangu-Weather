# 实施验证记录

验证日期：2026-09-19 UTC。

## 环境

- 系统 conda：`/home/hufeng/miniconda3/envs/pangu`，Python 3.10.21。
- GPU：NVIDIA GeForce RTX 4090 D，24 GB，驱动 575.51.03。
- NumPy 2.2.6、xarray 2025.6.1、NetCDF4 1.7.3、ONNX Runtime GPU 1.22.0、cdsapi 0.7.7。
- 完整 Python 依赖版本见 `validation-environment.txt`；`pip check` 通过。
- 项目目录内没有安装依赖或建立虚拟环境。

实际验证发现：两个模型会话同时缓存时，24 GB GPU 在切换到 24 小时模型后显存不足。
因此默认 `max_sessions` 调整为 1，先释放上一会话再切换模型。后续所有真实 GPU 验证均采用此设置。
该调整不改变贪婪模型选择或输出时效，只增加模型重新加载的开销。

## 自动测试

- `python -m pytest -q`：35 项离线测试通过，1 项需要真实模型的集成测试默认跳过。
- `PANGU_RUN_INTEGRATION=1 python -m pytest tests/test_gpu_integration.py -q -s`：1 项通过，耗时 123.20 秒。
- GPU 对照使用 2025-09-22 00 UTC 的真实 ERA5：逐项比较 6、12、18、24、30 小时结果与独立复现的
  官方 24/6 小时迭代流程，全部满足 `rtol=1e-6, atol=1e-6`；另运行了 3/1 小时模型。
- 离线测试覆盖所有时效的贪婪链、动态规划最少步数对照、坐标和层次重排、多时间精确选取、
  输入错误、缓存修复、失败隔离、共享结果、截断产物重跑、配置及权重变化拒绝错误复用、无副作用预览。

pytest 会报告一条 NetCDF4 的 `numpy.ndarray size changed` 导入警告。
这是[上游已修复的检查问题](https://github.com/Unidata/netcdf4-python/pull/1471)，当前固定发布版本仍会报告；
本次保留警告，没有全局屏蔽它，数据回读和实际模型数值对照均通过。

## 真实 case 批次

使用 `configs/validation.yaml` 与 `configs/cases.validation.csv`：

| case | 起报时间 UTC | 预报时长 | 输出间隔 | 首次结果 |
|---|---|---:|---:|---|
| ragasa_00_30h | 2025-09-22 00:00 | 30 h | 6 h | 完成 |
| ragasa_00_5h | 2025-09-22 00:00 | 5 h | 1 h | 完成 |
| ragasa_00_30h_repeat | 2025-09-22 00:00 | 30 h | 6 h | 复用第一份结果 |
| ragasa_06_6h | 2025-09-22 06:00 | 6 h | 6 h | 自动补齐 ERA5 后完成 |

首次执行：3 complete、1 reused、0 failed。
再次使用 `--resume`：4 reused、0 failed，没有加载模型或执行推理。

结果保存在 `outputs/validation/`，汇总表为 `summary.csv`。三个独立预报的 NetCDF 合计
2,020,500,364 字节（约 2.02 GB）；重复案例通过相对链接共享同一文件。
35 项离线测试另外覆盖不能被输出间隔整除的时长，例如 56 小时 / 6 小时。

全部真实输出已逐变量、逐时效回读，确认时间关系正确、没有缺失或非有限值；
另一起报时间的 6 小时输出有效时刻为 2025-09-22 12 UTC。
没有残留 `.partial` 文件，也没有生成 NPY 中间文件。

这次验证确认工程流程和官方推理链的一致性，不代表已完成台风路径或强度预报技巧评估。
