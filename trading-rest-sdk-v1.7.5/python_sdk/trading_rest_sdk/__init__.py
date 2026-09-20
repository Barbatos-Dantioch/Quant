"""
Trading REST SDK - Python客户端
用于查询和下载金融市场数据
"""

__version__ = "1.7.5"
__author__ = "Trading Team"

from .client import TradingRestClient
from .exceptions import (
    TradingRestError,
    APIError,
    AuthenticationError,
    NotFoundError,
    TaskError,
)
from .models import (
    QueryResult,
    DownloadTask,
    TaskStatus,
    DataCatalog,
)

# 复用WebSocket SDK的数据模型（如果已安装）
try:
    from trading_websocket_sdk import (
        SZStockSnapshot,
        SHStockSnapshot,
        CFFEXFutureSnapshot,
        parse_message,
    )
    _HAS_WS_MODELS = True
except ImportError:
    _HAS_WS_MODELS = False
    SZStockSnapshot = None
    SHStockSnapshot = None
    CFFEXFutureSnapshot = None
    parse_message = None

__all__ = [
    "TradingRestClient",
    "QueryResult",
    "DownloadTask",
    "TaskStatus",
    "DataCatalog",
    "TradingRestError",
    "APIError",
    "AuthenticationError",
    "NotFoundError",
    "TaskError",
]
