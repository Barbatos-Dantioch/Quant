# REST SDK 配置说明

## ⚠️ 重要：API Key配置

所有示例代码都需要配置API Key才能运行。

## 📝 配置方式

### 方式1：使用环境变量（推荐，最安全）

#### Linux / Mac
```bash
# 设置API Key
export REST_API_KEY="your_api_key_here"

# 运行示例
python examples/simple_query.py
```

#### Windows
```cmd
# CMD
set REST_API_KEY=your_api_key_here
python examples/simple_query.py

# PowerShell
$env:REST_API_KEY="your_api_key_here"
python examples/simple_query.py
```

### 方式2：使用配置文件

```bash
# 1. 复制模板
cp examples/config_template.py examples/config.py

# 2. 编辑config.py，填写你的API Key
# REST_API_KEY = "your_actual_key"
# REST_BASE_URL = "http://192.168.20.10:8080"

# 3. 在代码中使用
from config import REST_API_KEY, REST_BASE_URL
client = TradingRestClient(api_key=REST_API_KEY, base_url=REST_BASE_URL)
```

### 方式3：直接在代码中填写（不推荐）

```python
# 在示例代码中取消注释并填写
API_KEY = "your_api_key_here"
```

## 🌐 服务器地址配置

### 内网环境（默认）
```python
client = TradingRestClient(
    api_key="xxx",
    base_url="http://192.168.20.10:8080"
)
```

### 外网环境
```python
client = TradingRestClient(
    api_key="xxx",
    base_url="http://61.151.241.233:8080"
)
```

## 🔒 安全建议

1. ⚠️ **永远不要**将API Key硬编码在代码中
2. ⚠️ **永远不要**将API Key提交到Git仓库
3. ✅ **使用环境变量**是最安全的方式
4. ✅ 如使用配置文件，确保`config.py`在`.gitignore`中
5. ✅ 定期轮换API Key
6. ✅ 给不同环境（开发/测试/生产）使用不同的Key

## 📋 权限说明

REST API需要以下权限：

### 基础权限
- `rest.query` - 查询数据
- `rest.catalog` - 获取数据目录

### 下载权限
- `rest.download.task.create` - 创建下载任务
- `rest.download.task.status` - 查询任务状态
- `rest.download.file` - 下载文件

### PostgreSQL权限
- `rest.dbdict.download` - 下载表数据

请确保你的API Key具有相应权限。
