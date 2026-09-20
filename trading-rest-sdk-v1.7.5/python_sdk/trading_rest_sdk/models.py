"""
REST SDK数据模型
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueryResult:
    """查询结果"""
    key: str
    data: Dict[str, Any]
    timestamp: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DownloadTask:
    """下载任务"""
    task_id: str
    status: TaskStatus
    progress: int = 0
    message: str = ""
    file_id: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    record_count: Optional[int] = None
    download_url: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadTask":
        """从API响应创建任务对象"""
        status_str = data.get("status", "pending")
        status = TaskStatus(status_str) if status_str in [s.value for s in TaskStatus] else TaskStatus.PENDING
        
        # 从result字段提取文件信息
        result = data.get("result", {}) or {}
        
        return cls(
            task_id=data.get("id") or data.get("task_id", ""),
            status=status,
            progress=data.get("progress", 0),
            message=data.get("message", ""),
            file_id=result.get("file_id") or data.get("file_id"),
            file_name=result.get("file_name") or data.get("file_name"),
            file_size=result.get("file_size") or data.get("file_size"),
            record_count=result.get("record_count") or data.get("record_count"),
            download_url=result.get("download_url") or data.get("download_url"),
            error=data.get("error"),
        )


@dataclass
class DataCatalog:
    """数据目录"""
    version: str
    updated: str
    data_types: List[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataCatalog":
        """从API响应创建目录对象"""
        return cls(
            version=data.get("version", ""),
            updated=data.get("updated", ""),
            data_types=data.get("data_types", []),
        )


@dataclass
class RAWStats:
    """RAW数据统计信息"""
    message_type: str
    date: str
    key_count: int
    message_count: int
    first_key: Optional[str] = None
    last_key: Optional[str] = None
