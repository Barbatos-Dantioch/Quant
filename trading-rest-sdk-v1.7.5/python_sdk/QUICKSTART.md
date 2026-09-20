# 快速入门指南

## 5分钟上手 Trading REST SDK

### 第一步：安装

```bash
cd python_sdk
pip install -r requirements.txt
pip install -e .
```

### 第二步：准备API Key

设置环境变量：

```bash
export REST_API_KEY=your_api_key_here
```

### 第三步：运行第一个示例

```bash
python examples/simple_query.py
```

### 第四步：尝试下载数据

```bash
# 下载行情数据
python examples/download_decoded.py

# 下载静态数据
python examples/download_postgres.py
```

### 第五步：学习pandas分析

```bash
# 先安装pandas
pip install pandas

# 运行教程
python examples/pandas_tutorial.py
```

## 📝 基本使用

### 查询数据

```python
from trading_rest_sdk import TradingRestClient

client = TradingRestClient(api_key="your_key")

# 查询单条
data = client.query_decoded("ZZ-01", "SZ.000001", "20250930", "0930")
```

### 下载数据

```python
# DECODED行情数据
csv_file = client.download_decoded_csv(
    message_type="ZZ-01",
    symbols=["SZ.000001"],
    date_range={"start": "20250901", "end": "20250930"}
)

# PostgreSQL表数据
csv_file = client.download_postgres_table(
    table="mkt_equ_perf",
    format="csv"
)
```

### 使用pandas分析

```python
import pandas as pd

# 读取CSV
df = pd.read_csv(csv_file)

# 分析
print(df.head())
print(df['涨跌幅'].mean())
```

## ❓ 常见问题

**Q: 如何获取API Key？**  
A: 请联系服务提供商申请。

**Q: 内网和外网地址有什么区别？**  
A: 内网：`http://192.168.20.10:8080`，外网：`http://61.151.241.233:8080`

**Q: 能下载RAW原始数据吗？**  
A: 不能。RAW是二进制格式，只能查询统计信息。请使用DECODED数据。

**Q: 需要安装pandas吗？**  
A: 不强制。SDK可以下载CSV/JSON，pandas用于数据分析（可选）。

**Q: 710张表都有哪些？**  
A: 使用 `client.list_tables()` 查看所有可用表。

## 🎯 下一步

- 查看完整API文档：`README.md`
- 学习pandas分析：`examples/pandas_tutorial.py`
- 探索更多功能：运行其他示例代码
