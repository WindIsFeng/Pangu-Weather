# Pangu-Weather 本地部署版本

## 简介

本仓库提供了华为开发的 **Pangu-Weather** 天气预报模型的本地部署版本。Pangu-Weather 是一个基于 3D 神经网络的全球中期天气预报模型，能够在高分辨率（0.25°）下快速生成准确的天气预报，支持 1 小时、3 小时、6 小时和 24 小时的预测。

该模型最初发布于 [Nature 论文](https://www.nature.com/articles/s41586-023-06185-3)（2023 年）和 [arXiv 预印本](https://arxiv.org/abs/2206.11978)（2022 年）。本仓库基于官方实现（[官方 GitHub](https://github.com/198808xc/Pangu-Weather)）进行了本地部署优化，包括简化安装脚本和示例运行环境。

### 特性
- **高精度预报**：在 Z500、T850 等关键变量上优于传统数值天气预报模型（如 IFS）。
- **高效推理**：支持 CPU 和 GPU 加速，单次 24 小时预报可在几分钟内完成。
- **易部署**：提供简易部署，便于本地或服务器环境运行。
- **数据兼容**：支持 ERA5 或 ECMWF 数据输入。

## 环境要求
- Python 3.10
- 对于 GPU 支持：NVIDIA CUDA 12 + cuDNN 9
- NumPy、ONNX Runtime 等依赖（详见安装部分）
- 输入数据：ERA5 数据

## 安装

### 1. 克隆仓库
```bash
git clone git@github.com:WindIsFeng/Pangu.git
```

### 2. 安装依赖

- **GPU 环境**（确保 CUDA 已安装）：
  ```bash
  conda env create -f pangu.yml
  ```

依赖包括：`numpy`, `onnxruntime`（GPU 版）, `netCDF4`, `scipy` 等。完整列表见 `pangu.yml`。

### 3. 下载预训练模型
模型文件（每个约 1.1 GB）可从以下链接下载，放置到 `models/` 目录下：
- 1 小时模型：`pangu_weather_1.onnx` [Google Drive](https://drive.google.com/file/d/1fg5jkiN_5dHzKb-5H9Aw4MOmfILmeY-S/view?usp=share_link)
- 3 小时模型：`pangu_weather_3.onnx` [Google Drive](https://drive.google.com/file/d/1EdoLlAXqE9iZLt9Ej9i-JW9LTJ9Jtewt/view?usp=share_link)
- 6 小时模型：`pangu_weather_6.onnx` [Google Drive](https://drive.google.com/file/d/1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4/view?usp=share_link)
- 24 小时模型：`pangu_weather_24.onnx` [Google Drive](https://drive.google.com/file/d/1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP/view?usp=share_link)

（备选：Baidu Netdisk 链接见官方仓库）

## 使用指南

### 输入数据准备
Pangu-Weather 需要两个 .npy 文件作为输入：
- `input_surface.npy`：形状 (4, 721, 1440)，包含表面变量（MSLP, U10, V10, T2M）。
- `input_upper.npy`：形状 (5, 13, 721, 1440)，包含上层大气变量（Z, Q, T, U, V）在 13 个气压层（1000hPa 到 50hPa）。

使用提供的 `data_global.py` 脚本从 ERA5 .nc 文件转换：
```bash
python data_global.py
```

### 运行推理
- **预报**：
  ```bash
  python inference_global.py 
  ```
  输出将保存到 `output/`，包括预报变量的 .npy 文件。


- **可视化输出**（可选，使用 Matplotlib）：




## 参考文献
- Bi, K. et al. Accurate medium-range global weather forecasting with 3D neural networks. *Nature* 619, 533–538 (2023). [DOI: 10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3)


## 联系
- 问题反馈：通过 GitHub Issues 提交。
