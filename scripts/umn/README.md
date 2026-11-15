# UMN UAV 数据下载、CSV 整理与核验

在项目根目录使用本项目 `.venv`，所有数据集中在 `data/umn/`。

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\scripts\umn\requirements.txt
.\.venv\Scripts\python.exe -B .\scripts\umn\download_umn_uav.py --download --workers 12
.\.venv\Scripts\python.exe -B .\scripts\umn\convert_umn_to_csv.py --workers 4
.\.venv\Scripts\python.exe -B .\scripts\umn\verify_umn_csv.py
```

- `download_umn_uav.py`：公开接口检索、原始文件下载与官方 MD5/SHA-256 校验，可续传和复用已验证文件。
- `supervise_umn_download.py`：Windows 后台下载监控和失败重试；原始下载已完成，平时不需要运行。
- `probe_umn.py`：官方页面与 API 连通性诊断。
- `inspect_umn_data.py`、`inspect_special_cases.py`：本地文件结构、原始多采样率时序和数据字典检查。
- `convert_umn_to_csv.py`：一条官方记录一个 CSV，保留数据来源与原始字段；处理 ZIP、HDF5、MAT、Vireo 异步数据和导航修复版本。
- `verify_umn_csv.py`：全部文件的覆盖范围、行列数、BOM、SHA-256 核验，并对各机型及特殊结构进行原值往返比对。

`data/umn/raw/` 保存原始附件；`data/umn/csv/` 保存 CSV；`data/umn/conversion_metadata/` 保存每条记录的来源、字段和非时序元数据。转换规则、数量和特殊情况见 `data/umn/CSV_README.md`。

CSV 是原始数据整理结果，不等同于已完成阶段识别、清洗、插值或训练样本构造的数据。特别是 Vireo，不同传感器时钟的精确外连接会产生空值；需要后续根据任务选择时间对齐方案。
