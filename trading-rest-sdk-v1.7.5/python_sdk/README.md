# Trading REST SDK v1.6.0

> Python SDK for Trading REST API - 金融市场数据查询和下载客户端

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.6.0-orange)](setup.py)

---

## 📚 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [完整API参考](#完整api参考)
- [常见问题](#常见问题)

---

## ✨ 特性

### v1.6.0 新特性 🆕

- 🎯 **数据字典API** - 浏览行情数据源、PostgreSQL表、ClickHouse表
- 🔍 **智能SQL查询** - 自动识别数据库，自动处理字段名
- 📊 **表预览搜索** - 快速预览和搜索功能
- 🎛️ **任务管理** - 列出、取消下载任务
- 🔧 **SQL构建器** - 自动生成SQL，避免字段名错误

### 核心功能

- 💾 **多数据源支持** - Redis行情 + PostgreSQL静态数据 + ClickHouse历史数据（根据用户权限动态访问）
- 🔥 **SQL查询** - 自动识别PostgreSQL/ClickHouse，支持复杂查询
- 📊 **异步下载** - 支持大表下载（百万级数据）
- 🔐 **认证管理** - API Key自动处理
- 📝 **完善文档** - 详细的使用说明和故障排查

---

## 🚀 快速开始

### 安装

```bash
# 解压SDK
tar xzf trading-rest-sdk-v1.6.0-客户交付包.tar.gz
cd python_sdk

# 安装依赖
pip install -r requirements.txt

# 安装SDK
pip install -e .
```

### 第一个程序

```python
from trading_rest_sdk import TradingRestClient

# 创建客户端
client = TradingRestClient(
    api_key="your_api_key_here",
    base_url="http://61.151.241.233:8080"  # 公网地址
)

# 查询行情数据
data = client.query_decoded(
    message_type="ZZ-01",
    symbol="SZ.000001",
    date="20251107",
    minute="0930"
)

print(f"查询成功: {data}")
```

---

## 🎯 核心功能

### 1. 数据字典（v1.6.0新增）

**浏览所有数据源和表：**

```python
# 获取市场列表
markets = client.get_markets()
for market in markets:
    print(f"{market['name']}: {market['source_count']}个数据源")

# 列出所有数据源（根据用户权限动态返回）
sources = client.list_data_sources()
print(f"共{len(sources)}个数据源")

# 获取字段定义
fields = client.get_fields("ZZ-7001")  # 指数K线
for field in fields:
    print(f"{field['name_en']} - {field['name_cn']}")

# 预览指数K线数据
preview = client.preview_data("ZZ-7001")
print(f"预览数据: {len(preview)}条")

# 预览数据源数据
data = client.preview_data("ZZ-111", limit=10)
print(f"预览{len(data)}条数据")

# 搜索数据源
results = client.search_data_sources("K线")
print(f"找到{len(results)}个相关数据源")
```

### 2. SQL查询（推荐方式）🆕

**方式1：使用build_sql自动生成（最推荐）**

```python
# SDK自动处理PostgreSQL字段名双引号问题
sql_obj = client.build_sql(
    table="mkt_equd",
    columns=["TICKER_SYMBOL", "CLOSE_PRICE", "TRADE_DATE"],
    conditions={"TRADE_DATE": "2025-11-06"},
    order_by="CLOSE_PRICE DESC",
    limit=100
)

# 查看生成的SQL
print(sql_obj['sql'])
# 输出: SELECT "TICKER_SYMBOL", "CLOSE_PRICE", "TRADE_DATE" 
#       FROM mkt_equd WHERE "TRADE_DATE" = '2025-11-06' 
#       ORDER BY "CLOSE_PRICE" DESC LIMIT 100;

# 执行查询（自动识别数据库）
result = client.execute_sql(sql_obj['sql'])
print(f"查询结果: {result['count']}条")
```

**方式2：手写SQL（需注意字段名）**

```python
# PostgreSQL字段名大小写敏感，必须加双引号
result = client.execute_sql(
    sql='SELECT "TICKER_SYMBOL", "CLOSE_PRICE" FROM mkt_equd WHERE "TRADE_DATE" = \'2025-11-06\' LIMIT 10'
)

# ClickHouse不需要双引号
result = client.execute_sql(
    sql="SELECT stock_code, close_price FROM zz_5001 WHERE zzDate = '20251107' LIMIT 10"
)

# 下载CSV格式
csv_file = client.execute_sql(
    sql='SELECT * FROM mkt_equd WHERE "TRADE_DATE" = \'2025-11-06\' LIMIT 100',
    format="csv"
)
print(f"CSV已保存: {csv_file}")
```

### 3. DECODED行情数据查询下载

**单条查询：**

```python
# 查询单只股票某个时刻的快照
data = client.query_decoded(
    message_type="ZZ-01",
    symbol="SZ.000001",
    date="20251107",
    minute="0930"
)
```

**异步下载任务：**

```python
# 创建下载任务
task_id = client.create_download_task(
    message_type="ZZ-01",
    date_range={"start": "20251106", "end": "20251107"},
    symbols=["SZ.000001", "SZ.000002"],  # 可选
    time_range={"start": "0930", "end": "1500"},  # 可选
    fields=["证券代码", "最新价", "成交量"],  # 可选，中文字段名
    format="csv"
)

# 等待任务完成
task = client.wait_for_task(task_id, timeout=300)

# 下载文件
file_path = client.download_file(task.file_id)
print(f"下载完成: {file_path}")
```

### 4. PostgreSQL表数据下载

**方式1：快捷方法（推荐）**

```python
# 下载CSV
csv_file = client.download_postgres_csv(
    table="mkt_equd",
    columns=["TICKER_SYMBOL", "CLOSE_PRICE", "TRADE_DATE"],  # 可选
    date_range={"start_date": "2025-11-06", "end_date": "2025-11-07"},  # 可选
    limit=1000  # 可选（注意：API设计为全表下载，limit可能不生效）
)
print(f"下载完成: {csv_file}")
```

**方式2：通用方法**

```python
# 下载JSON格式（带进度回调）
def on_progress(progress, message):
    print(f"进度: {progress}% - {message}")

data = client.download_postgres_table(
    table="mkt_equd",
    date_range={"start_date": "2025-11-06", "end_date": "2025-11-07"},
    format="json",
    progress_callback=on_progress
)
```

**预览表数据：**

```python
# 预览表数据（最新10条）
data = client.preview_table("mkt_equd", limit=10)
print(f"预览{len(data)}条数据")

# 搜索表
tables = client.search_tables("股票")
print(f"找到{len(tables)}张相关表")
```

### 5. ClickHouse数据查询

```python
# ClickHouse查询（自动识别）
result = client.execute_sql(
    sql="SELECT * FROM zz_5001 WHERE zzDate = '20251107' LIMIT 100"
)

# 或者使用build_sql
sql_obj = client.build_sql(
    table="zz_5001",
    columns=["stock_code", "close_price"],
    conditions={"zzDate": "20251107"},
    limit=100,
    datasource="clickhouse"
)
result = client.execute_sql(sql_obj['sql'])
```

### 6. PostgreSQL表管理

```python
# 列出所有表
tables = client.list_tables()
print(f"共{len(tables)}张表")

# 获取表结构
schema = client.get_table_schema("mkt_equd")
print(f"表注释: {schema['comment']}")
print(f"字段数: {len(schema['columns'])}")
```

### 7. 任务管理（v1.6.0新增）

```python
# 列出所有任务
tasks = client.list_tasks(limit=20)

# 只列出进行中的任务
processing_tasks = client.list_tasks(status="processing")

# 取消任务
success = client.cancel_task("task_id_here")
if success:
    print("任务已取消")
```

---

## 📖 完整API参考

### 数据字典API

| 方法 | 说明 | 版本 |
|------|------|------|
| `get_markets()` | 获取市场列表 | v1.6.0 |
| `list_data_sources()` | 列出所有数据源 | v1.6.0 |
| `get_fields(code)` | 获取字段定义 | v1.6.0 |
| `preview_data(code)` | 预览数据源数据 | v1.6.0 |
| `search_data_sources(keyword)` | 搜索数据源 | v1.6.0 |
| `get_dictionary(code)` | 获取数据字典 | v1.5.0 |
| `get_catalog()` | 获取数据目录 | v1.4.0 |

### SQL查询API

| 方法 | 说明 | 版本 |
|------|------|------|
| `build_sql(...)` | SQL构建器（自动加双引号） | v1.6.0 |
| `execute_sql(sql)` | 执行SQL（自动识别数据库） | v1.6.0 |

### DECODED行情数据API

| 方法 | 说明 | 版本 |
|------|------|------|
| `query_decoded(...)` | 查询单条数据 | v1.4.0 |
| `create_download_task(...)` | 创建下载任务 | v1.4.0 |
| `get_task_status(task_id)` | 查询任务状态 | v1.4.0 |
| `wait_for_task(task_id)` | 等待任务完成 | v1.4.0 |
| `download_file(file_id)` | 下载文件 | v1.4.0 |
| `download_decoded_csv(...)` | 下载CSV（便捷方法） | v1.5.0 |

### PostgreSQL表数据API

| 方法 | 说明 | 版本 |
|------|------|------|
| `list_tables()` | 列出所有表 | v1.5.0 |
| `get_table_schema(table)` | 获取表结构 | v1.5.0 |
| `preview_table(table)` | 预览表数据 | v1.6.0 |
| `search_tables(keyword)` | 搜索表 | v1.6.0 |
| `download_postgres_table(...)` | 下载表数据 | v1.5.0 |
| `download_postgres_csv(...)` | 下载CSV（便捷） | v1.6.0 |
| `download_clickhouse_csv(...)` | 下载ClickHouse CSV | v1.6.0 |

### 任务管理API

| 方法 | 说明 | 版本 |
|------|------|------|
| `list_tasks()` | 列出任务 | v1.6.0 |
| `cancel_task(task_id)` | 取消任务 | v1.6.0 |

---

## ❓ 常见问题

### Q1: PostgreSQL查询为什么报"字段不存在"？

A: PostgreSQL字段名大小写敏感，必须用双引号包裹。

**解决方案：使用build_sql()自动生成SQL**

```python
# ✅ 推荐：自动处理
sql_obj = client.build_sql(
    table="mkt_equd",
    columns=["TICKER_SYMBOL"],  # SDK自动加双引号
    limit=10
)
result = client.execute_sql(sql_obj['sql'])
```

### Q2: 如何知道表在PostgreSQL还是ClickHouse？

A: 不需要知道！`execute_sql()`会自动识别。

```python
# 自动识别数据库（推荐）
result = client.execute_sql("SELECT * FROM mkt_equd LIMIT 10")
```

### Q3: conditions参数如何使用？

A: 使用字典格式，SDK会自动处理。

```python
# 正确格式
client.download_postgres_csv(
    table="mkt_equd",
    conditions={"TRADE_DATE": "2025-11-06"}  # 字典
)
```

### Q4: limit参数为什么不生效？

A: PostgreSQL下载设计为**全表下载**（API设计），不限制行数。如需限制，使用SQL查询：

```python
# 用SQL查询限制行数
sql_obj = client.build_sql(table="mkt_equd", limit=100)
result = client.execute_sql(sql_obj['sql'])
```

### Q5: 如何下载带时间段的数据？

A: 使用time_range参数：

```python
task_id = client.create_download_task(
    message_type="ZZ-01",
    date_range={"start": "20251107", "end": "20251107"},
    time_range={"start": "0930", "end": "1130"},  # 9:30-11:30
    format="csv"
)
```

---

## 📦 多数据源支持

**⚠️ 重要：** 用户可访问的数据源根据API Key权限动态决定。使用 `get_datasources()` 查询当前用户的数据源权限。

### 1. Redis（实时行情库）

- **数据类型**：实时行情数据（深圳、上海、期货、期权等市场）
- **特点**：实时+历史，毫秒级延迟
- **使用**：`query_decoded()`, `create_download_task()`
- **示例**：ZZ-01（深圳股票）、ZZ-5001（股票K线）、ZZ-7001（指数K线）

### 2. PostgreSQL（财务数据库）

- **数据类型**：静态财务数据（宏观经济、财务报表、行业分类等）
- **特点**：全量静态数据，每日更新
- **使用**：`download_postgres_csv()`, `execute_sql(datasource="postgresql")`
- **查询表数量**：`len(client.list_tables(datasource="postgresql"))`

### 3. ClickHouse（数据加工库）

- **数据类型**：历史加工宽表
- **特点**：海量历史数据，秒级查询
- **使用**：`execute_sql(datasource="clickhouse")`, `download_clickhouse_csv()`
- **查询表数量**：`len(client.list_tables(datasource="clickhouse"))`

### 4. ClickHouse（行情镜像库）

- **数据类型**：行情历史数据（Redis持久化）
- **特点**：完整行情历史，按市场分类
- **使用**：`execute_sql(datasource="clickhouse_data")`, `download_table_csv(datasource="clickhouse_data")`
- **查询表数量**：`len(client.list_tables(datasource="clickhouse_data"))`

**动态查询用户可访问的数据源：**
```python
# 获取当前用户的数据源权限
datasources = client.get_datasources()
for ds in datasources:
    status = "✅ 可访问" if ds['available'] else "❌ 无权限"
    print(f"{status} {ds['name']}: {ds['tables']}张表")
```

---

## 🔧 初始化配置

```python
from trading_rest_sdk import TradingRestClient

client = TradingRestClient(
    api_key="your_api_key",              # 必填：API密钥
    base_url="http://61.151.241.233:8080",  # REST API服务器（公网）
    # base_url="http://192.168.20.10:8080", # 内网地址
    timeout=60                            # 请求超时（秒）
)
```

---

## 📝 完整示例

### 示例1：查询并分析股票数据

```python
from trading_rest_sdk import TradingRestClient
import pandas as pd

client = TradingRestClient(api_key="your_key")

# 使用SQL查询
sql_obj = client.build_sql(
    table="mkt_equd",
    columns=["TICKER_SYMBOL", "CLOSE_PRICE", "TRADE_DATE"],
    conditions={"TRADE_DATE": "2025-11-06"},
    order_by="CLOSE_PRICE DESC",
    limit=100
)

result = client.execute_sql(sql_obj['sql'])

# 转为DataFrame分析
df = pd.DataFrame(result['data'])
print(f"平均价: {df['CLOSE_PRICE'].astype(float).mean():.2f}")
print(f"最高价: {df['CLOSE_PRICE'].astype(float).max():.2f}")
```

### 示例2：下载历史行情数据

```python
# 下载多只股票的历史快照
task_id = client.create_download_task(
    message_type="ZZ-01",
    date_range={"start": "20251101", "end": "20251107"},
    symbols=["SZ.000001", "SZ.000002", "SZ.000004"],
    time_range={"start": "0930", "end": "1500"},
    format="csv"
)

# 等待完成并下载
task = client.wait_for_task(task_id)
file_path = client.download_file(task.file_id)
print(f"下载完成: {file_path}, 大小: {task.file_size / 1024 / 1024:.1f}MB")
```

### 示例3：浏览和搜索数据

```python
# 搜索K线数据源
sources = client.search_data_sources("K线")
for src in sources:
    print(f"{src['code']}: {src['name']}")

# 预览数据
data = client.preview_data("ZZ-5001", limit=5)
print(f"K线数据示例: {data[0] if data else 'empty'}")

# 搜索包含"股票"的表
tables = client.search_tables("股票")
print(f"找到{len(tables)}张相关表")
```

---

## 🆚 REST SDK vs WebSocket SDK

| 特性 | REST SDK | WebSocket SDK |
|------|---------|---------------|
| **用途** | 历史数据查询下载 | 实时数据订阅 |
| **数据源** | Redis + PostgreSQL + ClickHouse | Redis实时推送 |
| **延迟** | 秒级（HTTP请求） | 毫秒级（推送） |
| **适合场景** | 回测、分析、研究 | 实盘、监控、交易 |
| **数据量** | 大批量下载 | 实时流式 |

**两个SDK可以配合使用：**
- REST SDK下载历史数据 → 策略回测
- WebSocket SDK订阅实时数据 → 实盘交易

---

## 📞 获取帮助

- 📖 **快速开始**：查看 `QUICKSTART.md`
- 📋 **版本历史**：查看 `CHANGELOG.md`
- 📝 **示例代码**：查看 `examples/` 目录
- 🐛 **问题反馈**：联系技术支持

---

## 📄 许可证

MIT License

---

> 💡 **提示**：PostgreSQL字段名大小写敏感，推荐使用 `build_sql()` 自动生成SQL！
