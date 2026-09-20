"""
配置文件模板

请复制此文件为 config.py，并填写你自己的配置信息。

使用方法：
1. 复制此文件：cp config_template.py config.py
2. 编辑 config.py，填写你的实际配置
3. 在代码中导入：from config import REST_API_KEY, REST_BASE_URL
"""

# ============================================================
# REST API服务器配置
# ============================================================

# REST API服务器地址
# 外网（默认）：http://61.151.241.233:8080
# 内网：http://192.168.20.10:8080
REST_BASE_URL = "http://61.151.241.233:8080"


# ============================================================
# API认证配置
# ============================================================

# API密钥
# ⚠️ 重要：请填写你自己的API Key！
# ⚠️ 不要将真实的API Key提交到版本控制系统！
REST_API_KEY = "your_api_key_here"


# ============================================================
# 下载配置
# ============================================================

# 下载文件保存目录
DOWNLOAD_DIR = "./downloads"

# 请求超时时间（秒）
REQUEST_TIMEOUT = 30

# 下载任务轮询间隔（秒）
TASK_POLL_INTERVAL = 2

# 下载任务最大等待时间（秒）
TASK_MAX_WAIT = 300


# ============================================================
# 使用示例
# ============================================================

"""
在你的代码中使用：

from config import REST_API_KEY, REST_BASE_URL
from trading_rest_sdk import TradingRestClient

client = TradingRestClient(
    api_key=REST_API_KEY,
    base_url=REST_BASE_URL
)
"""
