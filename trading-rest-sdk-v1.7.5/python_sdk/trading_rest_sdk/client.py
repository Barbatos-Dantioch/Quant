"""
Trading REST API 客户端
"""

import requests
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .exceptions import (
    APIError,
    AuthenticationError,
    NotFoundError,
    TaskError,
    ValidationError
)
from .models import QueryResult, DownloadTask, DataCatalog, RAWStats, TaskStatus


class TradingRestClient:
    """
    Trading REST API 客户端
    
    提供简洁的API用于查询和下载金融市场数据
    
    Example:
        ```python
        client = TradingRestClient(api_key="your_api_key")
        
        # 查询数据
        data = client.query_decoded("ZZ-01", "SZ.000001", "20250930", "0930")
        
        # 下载数据
        csv_file = client.download_table("mkt_equ_perf")
        ```
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://61.151.241.233:8080",
        timeout: int = 600,  # 默认10分钟，支持大表下载
        download_dir: str = "./downloads"
    ):
        """
        初始化REST客户端
        
        Args:
            api_key: API密钥
            base_url: API服务器地址
            timeout: 请求超时时间（秒）
            download_dir: 文件下载目录
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.download_dir = Path(download_dir)
        
        # 确保下载目录存在
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP会话
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        stream: bool = False
    ) -> requests.Response:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            params: URL参数
            json: JSON数据
            stream: 是否流式下载
        
        Returns:
            Response对象
        
        Raises:
            APIError: API请求失败
            AuthenticationError: 认证失败
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json,
                timeout=self.timeout,
                stream=stream
            )
            
            # 处理HTTP错误
            if response.status_code == 401:
                raise AuthenticationError("API Key认证失败")
            elif response.status_code == 404:
                raise NotFoundError(f"资源不存在: {endpoint}")
            elif response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', response.text)
                except:
                    error_msg = response.text
                raise APIError(error_msg, status_code=response.status_code)
            
            return response
            
        except requests.exceptions.Timeout:
            raise APIError(f"请求超时（{self.timeout}秒）")
        except requests.exceptions.ConnectionError:
            raise APIError(f"无法连接到服务器: {self.base_url}")
        except (AuthenticationError, NotFoundError, APIError):
            raise
        except Exception as e:
            raise APIError(f"请求失败: {e}")
    
    # ========== DECODED数据查询 ==========
    
    def query_decoded(
        self,
        message_type: str,
        symbol: str,
        date: str,
        minute: str,
        return_values: bool = True
    ) -> Dict[str, Any]:
        """
        查询DECODED行情数据
        
        Args:
            message_type: 消息类型，如 "ZZ-01"
            symbol: 股票代码，如 "SZ.000001"
            date: 日期，如 "20250930"（8位数字）
            minute: 时间，如 "0930"（4位数字）
            return_values: 是否返回数据值（默认True）
        
        Returns:
            查询结果字典
        
        Example:
            ```python
            result = client.query_decoded("ZZ-01", "SZ.000001", "20250930", "0930")
            print(f"总键数: {result['data']['total']}")
            print(f"示例数据: {result['data']['sample_data']}")
            ```
        """
        # 参数格式验证（只验证基本格式，不限制具体范围）
        self._validate_message_type(message_type)
        self._validate_date(date)
        self._validate_time(minute)
        # symbol不验证，格式太灵活，交给后端处理
        
        # 使用新版查询接口（POST /api/v1/query）
        payload = {
            "data_type": "DECODED",
            "message_type": message_type,
            "symbol": symbol,
            "date": date,
            "time": minute,
            "return_values": return_values,
            "limit": 1000
        }
        
        response = self._request("POST", "/api/v1/query", json=payload)
        return response.json()
    
    
    # ========== DECODED数据下载（异步任务）==========
    
    def create_download_task(
        self,
        message_type: str,
        date_range: Dict[str, str],
        symbols: Optional[List[str]] = None,
        time_range: Optional[Dict[str, str]] = None,
        fields: Optional[List[str]] = None,
        format: str = "csv"
    ) -> str:
        """
        创建DECODED数据下载任务
        
        Args:
            message_type: 消息类型，如 "ZZ-01"
            date_range: 日期范围 {"start": "20250901", "end": "20250930"}
            symbols: 股票代码列表（可选）
            time_range: 时间范围 {"start": "0930", "end": "1500"}（可选）
            format: 文件格式 "csv" 或 "json"
        
        Returns:
            任务ID
        
        Example:
            ```python
            task_id = client.create_download_task(
                message_type="ZZ-01",
                date_range={"start": "20250901", "end": "20250930"},
                symbols=["SZ.000001", "SZ.000002"],
                format="csv"
            )
            ```
        """
        # 后端使用下划线命名（符合DownloadTaskRequest结构）
        payload = {
            "data_type": "DECODED",
            "message_type": message_type,
            "date_start": date_range.get("start", ""),
            "date_end": date_range.get("end", ""),
            "format": format
        }
        
        if symbols:
            payload["symbols"] = symbols
        if time_range:
            payload["time_start"] = time_range.get("start", "")
            payload["time_end"] = time_range.get("end", "")
        if fields:
            payload["fields"] = fields
        
        response = self._request("POST", "/api/v1/download/task", json=payload)
        result = response.json()
        return result.get("task_id")
    
    def get_task_status(self, task_id: str) -> DownloadTask:
        """
        查询下载任务状态
        
        Args:
            task_id: 任务ID
        
        Returns:
            任务对象
        
        Example:
            ```python
            task = client.get_task_status(task_id)
            print(f"进度: {task.progress}%")
            print(f"状态: {task.status.value}")
            ```
        """
        response = self._request("GET", f"/api/v1/download/task/{task_id}")
        data = response.json()
        return DownloadTask.from_dict(data)
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: int = 300,
        poll_interval: int = 2
    ) -> DownloadTask:
        """
        等待任务完成
        
        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）
            poll_interval: 轮询间隔（秒）
        
        Returns:
            完成的任务对象
        
        Raises:
            TaskError: 任务失败或超时
        """
        start_time = time.time()
        
        while True:
            task = self.get_task_status(task_id)
            
            if task.status == TaskStatus.COMPLETED:
                return task
            elif task.status == TaskStatus.FAILED:
                raise TaskError(f"任务失败: {task.error}")
            elif task.status == TaskStatus.CANCELLED:
                raise TaskError("任务已取消")
            
            # 检查超时
            if time.time() - start_time > timeout:
                raise TaskError(f"任务超时（{timeout}秒）")
            
            # 等待后继续轮询
            time.sleep(poll_interval)
    
    def download_file(
        self,
        file_id: str,
        save_to: Optional[str] = None,
        task_type: str = "auto"
    ) -> str:
        """
        下载文件（智能路由）
        
        Args:
            file_id: 文件ID（从任务结果获取）
            save_to: 保存路径（可选）
            task_type: 任务类型 ("auto"自动识别/"decoded"行情任务/"dbdict"数据库任务)
        
        Returns:
            文件路径
        
        Example:
            ```python
            # 等待任务完成
            task = client.wait_for_task(task_id)
            
            # 下载文件（自动识别路径）
            file_path = client.download_file(task.file_id)
            print(f"文件已保存到: {file_path}")
            ```
        """
        # 智能选择下载路径
        if task_type == "auto":
            # 先尝试dbdict路径（新的静态数据下载）
            try:
                response = self._request(
                    "GET",
                    f"/api/v1/dbdict/download-file/{file_id}",
                    stream=True
                )
            except:
                # 如果失败，尝试DECODED路径（旧的行情数据下载）
                response = self._request(
                    "GET",
                    f"/api/v1/download/file/{file_id}",
                    stream=True
                )
        elif task_type == "dbdict":
            response = self._request(
                "GET",
                f"/api/v1/dbdict/download-file/{file_id}",
                stream=True
            )
        else:  # decoded
            response = self._request(
                "GET",
                f"/api/v1/download/file/{file_id}",
                stream=True
            )
        
        # 确定文件名
        if save_to:
            file_path = Path(save_to)
        else:
            # 从响应头获取文件名
            content_disp = response.headers.get('Content-Disposition', '')
            filename = file_id
            if 'filename=' in content_disp:
                filename = content_disp.split('filename=')[1].strip('"')
            file_path = self.download_dir / filename
        
        # 写入文件
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return str(file_path)
    
    # ========== PostgreSQL表数据下载 ==========
    
    def download_postgres_table(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        date_range: Optional[Dict[str, str]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        format: str = "csv",
        datasource: str = "postgresql",
        save_to: Optional[str] = None,
        poll_interval: int = 1,
        max_wait: int = 300,
        progress_callback: Optional[callable] = None
    ) -> Union[str, Dict]:
        """
        下载数据库表数据（异步方式，支持大表）
        
        Args:
            table: 表名
            columns: 字段列表（可选，空则下载所有字段）
            conditions: 查询条件（可选，键值对）
            date_range: 日期范围（可选，只需start_date和end_date，后端自动用UPDATE_TIME筛选）
            order_by: 排序字段（可选）
            format: "csv" 或 "json"
            datasource: 数据源 "postgresql" 或 "clickhouse"（默认postgresql）
            save_to: 保存路径（可选）
            poll_interval: 轮询间隔（秒），默认1秒
            max_wait: 最大等待时间（秒），默认300秒
            progress_callback: 进度回调函数 callback(progress, message)
        
        Returns:
            CSV格式：文件路径
            JSON格式：数据字典
        
        Example:
            ```python
            # 下载CSV（带日期范围）
            csv_file = client.download_postgres_table(
                table="block_trading",
                columns=["TRADE_DATE", "SECURITY_ID", "TRADE_PRICE"],
                date_range={
                    "start_date": "2025-08-01",
                    "end_date": "2025-08-31"
                },
                format="csv"
            )
            print(f"下载完成: {csv_file}")
            
            # 下载JSON（带条件筛选）
            data = client.download_postgres_table(
                table="block_trading",
                conditions={"CURRENCY_CD": "CNY"},
                date_range={
                    "start_date": "2025-08-29",
                    "end_date": "2025-08-29"
                },
                format="json"
            )
            
            # 带进度回调
            def on_progress(progress, message):
                print(f"进度: {progress}% - {message}")
            
            csv_file = client.download_postgres_table(
                table="block_trading",
                format="csv",
                progress_callback=on_progress
            )
            ```
        """
        # 1. 创建下载任务
        payload = {
            "table_name": table,
            "format": format
        }
        
        if columns:
            payload["columns"] = columns
        if conditions:
            payload["conditions"] = conditions
        if date_range:
            # 简化的date_range，只需start_date和end_date
            payload["date_range"] = {
                "start_date": date_range.get("start_date", ""),
                "end_date": date_range.get("end_date", "")
            }
        if order_by:
            payload["order_by"] = order_by
        if limit:
            payload["limit"] = limit
        
        if progress_callback:
            progress_callback(0, "创建下载任务...")
        
        # 创建任务（添加datasource参数）
        endpoint = "/api/v1/dbdict/download-task"
        params = {}
        if datasource != "postgresql":
            params["datasource"] = datasource
        
        response = self._request(
            "POST",
            endpoint,
            params=params,
            json=payload
        )
        
        result = response.json()
        if result.get("code") != 200:
            raise APIError(f"创建下载任务失败: {result.get('message', 'Unknown error')}")
        
        task_id = result["data"]["task_id"]
        
        if progress_callback:
            progress_callback(10, f"任务已创建: {task_id}")
        
        # 2. 轮询任务状态
        start_time = time.time()
        while True:
            # 检查超时
            if time.time() - start_time > max_wait:
                raise TaskError(f"任务超时（{max_wait}秒）")
            
            # 查询任务状态
            task_response = self._request(
                "GET",
                f"/api/v1/dbdict/download-task/{task_id}"
            )
            
            task_data = task_response.json()
            if task_data.get("code") != 200:
                raise APIError(f"查询任务状态失败: {task_data.get('message')}")
            
            task = task_data["data"]
            status = task["status"]
            progress = task.get("progress", 0)
            message = task.get("message", "")
            
            if progress_callback:
                progress_callback(progress, message)
            
            # 检查任务状态
            if status == "completed":
                # 任务完成，下载文件
                file_id = task["result"]["file_id"]
                file_name = task["result"]["file_name"]
                record_count = task["result"]["record_count"]
                
                if progress_callback:
                    progress_callback(100, f"任务完成，共{record_count}条数据，正在下载文件...")
                
                # 3. 下载文件
                file_response = self._request(
                    "GET",
                    f"/api/v1/dbdict/download-file/{file_id}",
                    stream=True
                )
                
                if format == "csv":
                    # 保存CSV文件
                    if save_to:
                        file_path = Path(save_to)
                    else:
                        file_path = self.download_dir / file_name
                    
                    with open(file_path, 'wb') as f:
                        for chunk in file_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    return str(file_path)
                else:
                    # 返回JSON数据
                    return file_response.json()
            
            elif status == "failed":
                error_msg = task.get("error", "Unknown error")
                raise TaskError(f"任务失败: {error_msg}")
            
            elif status == "cancelled":
                raise TaskError("任务已取消")
            
            # 任务还在处理中，等待后继续轮询
            time.sleep(poll_interval)
    
    # ========== RAW数据统计 ==========
    
    # ========== 元数据API ==========
    
    def get_catalog(self) -> DataCatalog:
        """
        获取数据目录
        
        Returns:
            数据目录对象
        
        Example:
            ```python
            catalog = client.get_catalog()
            for data_type in catalog.data_types:
                print(f"{data_type['pattern']}: {data_type['description']}")
            ```
        
        Note:
            如果后端返回占位符响应，会返回空目录。
            建议直接使用已知的消息类型（ZZ-01 ~ ZZ-107）。
        """
        response = self._request("GET", "/api/v1/catalog")
        data = response.json()
        
        # 检查是否是占位符响应
        if isinstance(data, dict) and 'message' in data and 'data_types' not in data:
            # 返回空目录，避免崩溃
            return DataCatalog(
                version="unknown",
                updated="",
                data_types=[]
            )
        
        return DataCatalog.from_dict(data)
    
    def get_dictionary(self, message_code: str, include_fields: bool = True) -> Dict[str, Any]:
        """
        获取数据字典（数据源详情 + 字段定义）
        
        Args:
            message_code: 消息编号，如 "ZZ-01"
            include_fields: 是否包含字段定义（默认True）
        
        Returns:
            完整的数据源信息（包含字段定义）
        
        Example:
            ```python
            dict_info = client.get_dictionary("ZZ-01")
            print(f"消息名称: {dict_info['name']}")
            print(f"市场: {dict_info['market']}")
            print(f"字段数量: {dict_info['field_count']}")
            
            # 遍历字段定义
            for field in dict_info['fields']:
                print(f"  {field['cn_name']}: {field['type']}")
            ```
        """
        # 1. 获取基本信息
        response = self._request("GET", f"/api/v1/dictionary/sources/{message_code}")
        result = response.json()
        
        # 提取data字段
        if isinstance(result, dict) and 'data' in result:
            info = result['data']
        else:
            info = result
        
        # 2. 获取字段定义（如果需要）
        if include_fields:
            try:
                fields_response = self._request("GET", f"/api/v1/dictionary/sources/{message_code}/fields")
                fields_result = fields_response.json()
                
                # 提取字段列表
                if isinstance(fields_result, dict) and 'data' in fields_result:
                    info['fields'] = fields_result['data']
                else:
                    info['fields'] = fields_result
                    
            except Exception as e:
                # 如果字段API失败，返回空列表，不影响基本信息
                info['fields'] = []
        
        return info
    
    def get_datasources(self) -> List[Dict[str, Any]]:
        """
        获取当前用户可访问的数据源列表（根据API Key权限）
        
        Returns:
            数据源列表，包含权限和可用性信息
        
        Example:
            ```python
            datasources = client.get_datasources()
            for ds in datasources:
                status = "✅" if ds['available'] else "❌"
                print(f"{status} {ds['name']}: {ds['tables']}张表")
                if not ds['has_permission']:
                    print(f"     无权限访问")
            ```
        """
        response = self._request("GET", "/api/v1/dbdict/datasources")
        result = response.json()
        if isinstance(result, dict) and 'data' in result:
            return result['data'].get('datasources', [])
        return []
    
    def list_tables(self, datasource: str = "postgresql") -> List[Dict[str, Any]]:
        """
        列出数据库表
        
        Args:
            datasource: 数据源 (postgresql/clickhouse/clickhouse_data)
        
        Returns:
            表列表
        
        Example:
            ```python
            # PostgreSQL表
            tables = client.list_tables()
            
            # ClickHouse数据加工库
            tables = client.list_tables(datasource="clickhouse")
            
            # ClickHouse行情镜像库
            tables = client.list_tables(datasource="clickhouse_data")
            
            for table in tables:
                print(f"{table['table_name']}: {table['table_comment']}")
            ```
        """
        params = {}
        if datasource != "postgresql":
            params["datasource"] = datasource
            
        response = self._request("GET", "/api/v1/dbdict/tables", params=params)
        result = response.json()
        # 后端返回格式：{"code": 200, "data": [...], "total": 710}
        # 提取data字段
        if isinstance(result, dict) and 'data' in result:
            return result['data']
        return result  # 兼容旧格式
    
    def get_table_schema(self, table: str, datasource: str = "postgresql") -> Dict[str, Any]:
        """
        获取表结构定义
        
        Args:
            table: 表名
            datasource: 数据源 (postgresql/clickhouse/clickhouse_data)
        
        Returns:
            表结构信息
        
        Example:
            ```python
            schema = client.get_table_schema("mkt_equ_perf")
            schema = client.get_table_schema("zz_01", datasource="clickhouse_data")
            for field in schema['fields']:
                print(f"{field['name']}: {field['type']}")
            ```
        """
        params = {}
        if datasource != "postgresql":
            params["datasource"] = datasource
        response = self._request("GET", f"/api/v1/dbdict/tables/{table}", params=params)
        return response.json()
    
    # ========== 便利方法 ==========
    
    def download_decoded_csv(
        self,
        message_type: str,
        symbols: List[str],
        date_range: Dict[str, str],
        save_to: Optional[str] = None
    ) -> str:
        """
        一站式下载DECODED数据为CSV
        
        自动创建任务、等待完成、下载文件
        
        Args:
            message_type: 消息类型
            symbols: 股票代码列表
            date_range: 日期范围
            save_to: 保存路径（可选）
        
        Returns:
            文件路径
        
        Example:
            ```python
            csv_file = client.download_decoded_csv(
                message_type="ZZ-01",
                symbols=["SZ.000001", "SZ.000002"],
                date_range={"start": "20250901", "end": "20250930"}
            )
            print(f"下载完成: {csv_file}")
            ```
        """
        # 创建任务
        task_id = self.create_download_task(
            message_type=message_type,
            date_range=date_range,
            symbols=symbols,
            format="csv"
        )
        
        # 等待任务完成
        task = self.wait_for_task(task_id)
        
        # 下载文件
        return self.download_file(task.file_id, save_to=save_to)
    
    def download_postgres_csv(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        date_range: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        save_to: Optional[str] = None,
        datasource: str = "postgresql"
    ) -> str:
        """
        下载数据库表为CSV（便捷方法）
        
        Args:
            table: 表名
            columns: 字段列表
            conditions: 查询条件
            date_range: 日期范围
            limit: 限制行数
            save_to: 保存路径
            datasource: 数据源 (postgresql/clickhouse/clickhouse_data)
        
        Returns:
            文件路径
        """
        return self.download_postgres_table(
            table=table,
            columns=columns,
            conditions=conditions,
            date_range=date_range,
            limit=limit,
            format="csv",
            datasource=datasource,
            save_to=save_to
        )
    
    def download_clickhouse_csv(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        date_range: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None,
        save_to: Optional[str] = None
    ) -> str:
        """
        下载ClickHouse表为CSV（便捷方法）
        
        Args:
            table: 表名
            columns: 字段列表
            conditions: 查询条件
            date_range: 日期范围
            limit: 限制行数
            save_to: 保存路径
        
        Returns:
            文件路径
        """
        return self.download_postgres_table(
            table=table,
            columns=columns,
            conditions=conditions,
            date_range=date_range,
            limit=limit,
            format="csv",
            datasource="clickhouse",
            save_to=save_to
        )
    
    def create_postgres_download_task(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        date_range: Optional[Dict[str, str]] = None,
        symbols: Optional[List[str]] = None,
        format: str = "csv",
        datasource: str = "postgresql"
    ) -> str:
        """
        创建数据库下载任务（异步）
        
        Args:
            table: 表名
            columns: 字段列表
            conditions: 查询条件
            date_range: 日期范围
            symbols: 股票代码列表
            format: 文件格式
            datasource: 数据源
        
        Returns:
            任务ID（用于查询状态和下载）
        """
        # 构建payload
        payload = {
            "table_name": table,
            "format": format
        }
        
        if columns:
            payload["columns"] = columns
        if conditions:
            payload["conditions"] = conditions
        if date_range:
            payload["date_range"] = date_range
        if symbols:
            payload["symbols"] = symbols
        if datasource:
            payload["datasource"] = datasource
        
        # 调用异步任务接口
        endpoint = "/api/v1/dbdict/download-task"
        params = {}
        if datasource and datasource != "postgresql":
            params["datasource"] = datasource
        
        response = self._request("POST", endpoint, params=params, json=payload)
        result = response.json()
        
        # 提取任务ID
        if isinstance(result, dict) and 'data' in result:
            return result['data']['task_id']
        return result.get('task_id', '')
    
    def create_clickhouse_download_task(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        date_range: Optional[Dict[str, str]] = None,
        symbols: Optional[List[str]] = None,
        format: str = "csv",
        datasource: str = "clickhouse"
    ) -> str:
        """
        创建ClickHouse下载任务（异步）
        
        Args:
            table: 表名
            columns: 字段列表
            conditions: 查询条件
            date_range: 日期范围
            symbols: 股票代码列表（筛选）
            format: 文件格式
            datasource: 数据源 (clickhouse/clickhouse_data)
        
        Returns:
            任务ID（用于查询状态和下载）
        """
        # 调用create_postgres_download_task（统一的异步任务接口）
        return self.create_postgres_download_task(
            table=table,
            columns=columns,
            conditions=conditions,
            date_range=date_range,
            symbols=symbols,
            format=format,
            datasource=datasource
        )
    
    # ========== SQL查询功能 ==========
    
    def execute_sql(
        self,
        sql: str,
        datasource: str = "",
        format: str = "json",
        timeout: int = 600  # 默认10分钟，支持大表查询
    ) -> Union[Dict, str]:
        """
        执行自定义SQL查询（仅支持SELECT）
        
        Args:
            sql: SQL查询语句（仅支持SELECT）
            datasource: 数据源（可选）
                       - 不指定：自动识别表所在数据库（推荐）
                       - "postgresql"：强制使用PostgreSQL
                       - "clickhouse"：强制使用ClickHouse数据加工库
                       - "clickhouse_data"：强制使用ClickHouse行情镜像库
            format: 返回格式 "json" 或 "csv"
            timeout: 查询超时时间（秒）
        
        Returns:
            JSON格式：查询结果字典
            CSV格式：文件路径
        
        Example:
            ```python
            # 推荐：不指定datasource，自动识别
            result = client.execute_sql(
                sql='SELECT "TICKER_SYMBOL", "CLOSE_PRICE" FROM mkt_equd LIMIT 10'
            )
            
            # 或者使用build_sql()自动生成SQL（更推荐）
            sql_obj = client.build_sql(
                table="mkt_equd",
                columns=["TICKER_SYMBOL", "CLOSE_PRICE"],
                limit=10
            )
            result = client.execute_sql(sql_obj['sql'])  # 自动识别数据库
            ```
        
        Note:
            - ✅ 推荐不指定datasource，让后端自动识别表所在数据库
            - ⚠️ PostgreSQL字段名大小写敏感，建议用build_sql()自动生成SQL
            - ClickHouse字段名不区分大小写
        """
        # 如果未指定datasource，尝试自动识别
        if not datasource:
            # 从SQL中提取表名
            import re
            table_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
            if table_match:
                table_name = table_match.group(1)
                
                # 先尝试PostgreSQL
                try:
                    tables_pg = self.list_tables()
                    if any(t.get('table_name') == table_name for t in tables_pg):
                        datasource = "postgresql"
                except:
                    pass
                
                # 再尝试ClickHouse数据加工库
                if not datasource:
                    try:
                        response = self._request("GET", "/api/v1/dbdict/tables", params={"datasource": "clickhouse", "size": 200})
                        tables_ch = response.json().get("data", [])
                        if any(t.get('table_name') == table_name for t in tables_ch):
                            datasource = "clickhouse"
                    except:
                        pass
                
                # 再尝试ClickHouse行情镜像库
                if not datasource:
                    try:
                        response = self._request("GET", "/api/v1/dbdict/tables", params={"datasource": "clickhouse_data", "size": 200})
                        tables_ch_data = response.json().get("data", [])
                        if any(t.get('table_name') == table_name for t in tables_ch_data):
                            datasource = "clickhouse_data"
                    except:
                        pass
            
            # 如果还是没识别出来，默认用postgresql
            if not datasource:
                datasource = "postgresql"
        
        payload = {
            "sql": sql,
            "datasource": datasource,
            "format": format,
            "timeout": timeout
        }
        
        response = self._request("POST", "/api/v1/dbdict/sql/query", json=payload)
        
        # CSV格式：直接保存文件
        if format == "csv":
            # 从响应头获取文件名
            content_disp = response.headers.get('Content-Disposition', '')
            filename = f"sql_query_{int(time.time())}.csv"
            if 'filename=' in content_disp:
                filename = content_disp.split('filename=')[1].strip('"')
            
            file_path = self.download_dir / filename
            
            # 保存CSV文件
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return str(file_path)
        
        # JSON格式：解析返回
        result = response.json()
        
        if result.get("code") != 200:
            raise APIError(f"SQL查询失败: {result.get('message', 'Unknown error')}")
        
        return result.get("data", {})
    
    def get_supported_databases(self) -> Dict[str, Any]:
        """
        获取支持的数据库列表
        
        Returns:
            数据库列表和SQL使用规则
        """
        response = self._request("GET", "/api/v1/dbdict/sql/databases")
        return response.json()
    
    # ========== 数据字典功能（v1.6.0新增）==========
    
    def get_markets(self) -> List[Dict[str, Any]]:
        """
        获取市场列表
        
        Returns:
            市场列表
        
        Example:
            ```python
            markets = client.get_markets()
            for market in markets:
                print(f"{market['name']}: {market['source_count']}个数据源")
            ```
        """
        response = self._request("GET", "/api/v1/dictionary/markets")
        result = response.json()
        return result.get("data", [])
    
    def list_data_sources(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出所有数据源
        
        Args:
            market: 市场筛选（可选）
        
        Returns:
            数据源列表
        
        Example:
            ```python
            # 获取所有数据源
            sources = client.list_data_sources()
            
            # 获取指定市场的数据源
            sources = client.list_data_sources(market="深圳")
            ```
        """
        params = {}
        if market:
            params["market"] = market
        
        response = self._request("GET", "/api/v1/dictionary/sources", params=params)
        result = response.json()
        return result.get("data", [])
    
    def get_fields(self, code: str) -> List[Dict[str, Any]]:
        """
        获取数据源字段定义
        
        Args:
            code: 数据源代码，如 "ZZ-111"
        
        Returns:
            字段列表
        
        Example:
            ```python
            fields = client.get_fields("ZZ-111")
            for field in fields:
                print(f"{field['name_en']} - {field['name_cn']}")
            ```
        """
        response = self._request("GET", f"/api/v1/dictionary/sources/{code}/fields")
        result = response.json()
        return result.get("data", [])
    
    def preview_data(self, code: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        预览数据源数据
        
        Args:
            code: 数据源代码，如 "ZZ-111"
            limit: 预览数量，默认10条
        
        Returns:
            数据列表
        
        Example:
            ```python
            data = client.preview_data("ZZ-111")
            print(f"预览了{len(data)}条数据")
            ```
        """
        params = {"limit": limit}
        response = self._request("GET", f"/api/v1/dictionary/sources/{code}/preview", params=params)
        result = response.json()
        return result.get("data", [])
    
    def search_data_sources(self, keyword: str) -> List[Dict[str, Any]]:
        """
        搜索数据源
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            匹配的数据源列表
        
        Example:
            ```python
            results = client.search_data_sources("K线")
            ```
        """
        params = {"keyword": keyword}
        response = self._request("GET", "/api/v1/dictionary/search", params=params)
        result = response.json()
        return result.get("data", [])
    
    # ========== PostgreSQL表功能增强（v1.6.0）==========
    
    def preview_table(self, table: str, limit: int = 10, datasource: str = "postgresql") -> List[Dict[str, Any]]:
        """
        预览数据库表数据
        
        Args:
            table: 表名
            limit: 预览数量，默认10条
            datasource: 数据源 (postgresql/clickhouse/clickhouse_data)
        
        Returns:
            数据列表
        
        Example:
            ```python
            data = client.preview_table("mkt_equd", limit=10)
            data = client.preview_table("zz_01", datasource="clickhouse_data")
            ```
        """
        params = {"limit": limit}
        if datasource != "postgresql":
            params["datasource"] = datasource
        response = self._request("GET", f"/api/v1/dbdict/tables/{table}/preview", params=params)
        result = response.json()
        return result.get("data", [])
    
    def search_tables(self, keyword: str, datasource: str = "postgresql") -> List[Dict[str, Any]]:
        """
        搜索数据库表和字段
        
        Args:
            keyword: 搜索关键词
            datasource: 数据源 (postgresql/clickhouse/clickhouse_data)
        
        Returns:
            匹配的表列表
        
        Example:
            ```python
            tables = client.search_tables("股票")
            tables = client.search_tables("zz", datasource="clickhouse_data")
            ```
        """
        params = {"keyword": keyword}
        if datasource != "postgresql":
            params["datasource"] = datasource
        response = self._request("GET", "/api/v1/dbdict/search", params=params)
        result = response.json()
        return result.get("data", [])
    
    def get_categories(self, datasource: str = "postgresql") -> List[Dict[str, Any]]:
        """
        获取数据库分类统计
        
        Args:
            datasource: 数据源 (postgresql/clickhouse/clickhouse_data)
        
        Returns:
            分类列表
        
        Example:
            ```python
            categories = client.get_categories()
            categories = client.get_categories(datasource="clickhouse_data")
            for cat in categories:
                print(f"{cat['name']}: {cat['table_count']}张表")
            ```
        """
        params = {}
        if datasource != "postgresql":
            params["datasource"] = datasource
        response = self._request("GET", "/api/v1/dbdict/categories", params=params)
        result = response.json()
        return result.get("data", [])
    
    def get_database_stats(self, datasource: str = "postgresql") -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Args:
            datasource: 数据源 (postgresql/clickhouse/clickhouse_data)
        
        Returns:
            统计信息
        
        Example:
            ```python
            stats = client.get_database_stats()
            print(f"总表数: {stats['total_tables']}")
            ```
        """
        params = {}
        if datasource != "postgresql":
            params["datasource"] = datasource
        response = self._request("GET", "/api/v1/dbdict/stats", params=params)
        result = response.json()
        return result.get("data", {})
    
    def build_sql(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        conditions: Optional[Union[Dict[str, Any], List[str]]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        datasource: str = "postgresql"
    ) -> Dict[str, Any]:
        """
        SQL构建器（自动生成SELECT语句）
        
        Args:
            table: 表名
            columns: 字段列表（可选，空则选择所有字段）
                     PostgreSQL会自动加双引号，无需手动添加
            conditions: 查询条件
                       - 字典格式：{"field": "value"} 会自动转为 ["field = 'value'"]（PostgreSQL自动加双引号）
                       - 列表格式：["field = 'value'", "price > 100"] 直接使用（需自己处理双引号）
            order_by: 排序字段（可选），PostgreSQL会自动加双引号
            limit: 限制行数（可选）
            datasource: 数据源，"postgresql"（默认）或"clickhouse"
        
        Returns:
            字典，包含：
            - sql: 生成的SQL语句
            - description: SQL说明
            - example: 示例SQL
        
        Example:
            ```python
            # PostgreSQL（自动加双引号，推荐）
            result = client.build_sql(
                table="mkt_equd",
                columns=["TICKER_SYMBOL", "CLOSE_PRICE"],  # SDK自动加双引号
                conditions={"TRADE_DATE": "2025-11-07"},   # SDK自动加双引号
                order_by="CLOSE_PRICE DESC",               # SDK自动加双引号
                limit=100,
                datasource="postgresql"
            )
            print(result['sql'])
            # 输出: SELECT "TICKER_SYMBOL", "CLOSE_PRICE" FROM mkt_equd 
            #       WHERE "TRADE_DATE" = '2025-11-07' ORDER BY "CLOSE_PRICE" DESC LIMIT 100;
            
            # ClickHouse（不需要双引号）
            result = client.build_sql(
                table="zz_5001",
                columns=["stock_code", "close_price"],
                conditions={"zzDate": "20251107"},
                limit=100,
                datasource="clickhouse"
            )
            ```
        
        Note:
            - PostgreSQL模式会自动给所有字段名加双引号
            - ClickHouse模式不加双引号
            - 建议使用此方法生成SQL，避免手写双引号错误
        """
        # PostgreSQL字段名需要加双引号，ClickHouse不需要
        quote_field = datasource == "postgresql"
        
        payload = {"table_name": table}
        
        # 处理columns（PostgreSQL自动加双引号）
        if columns:
            if quote_field:
                payload["columns"] = [f'"{col}"' if not col.startswith('"') else col for col in columns]
            else:
                payload["columns"] = columns
        
        # 处理conditions参数
        if conditions:
            if isinstance(conditions, dict):
                # 字典格式：转换为字符串数组
                cond_list = []
                for key, value in conditions.items():
                    # PostgreSQL字段名加双引号
                    field_name = f'"{key}"' if quote_field and not key.startswith('"') else key
                    
                    # 自动判断值的类型，添加引号
                    if isinstance(value, str):
                        cond_list.append(f"{field_name} = '{value}'")
                    else:
                        cond_list.append(f"{field_name} = {value}")
                payload["conditions"] = cond_list
            elif isinstance(conditions, list):
                # 列表格式：直接使用（假设用户已处理好）
                payload["conditions"] = conditions
        
        # 处理order_by（PostgreSQL字段名加双引号）
        if order_by:
            if quote_field and '"' not in order_by:
                # 简单处理：给字段名加双引号（处理 "field" 和 "field DESC"）
                parts = order_by.split()
                if len(parts) > 0:
                    field = parts[0]
                    direction = parts[1] if len(parts) > 1 else ""
                    order_by = f'"{field}" {direction}'.strip()
            payload["order_by"] = order_by
        
        if limit:
            payload["limit"] = limit
        
        response = self._request("POST", "/api/v1/dbdict/sql-builder", json=payload)
        result = response.json()
        
        # 返回完整的响应数据（包含sql、description、example等）
        return result.get("data", {})
    
    # ========== 任务管理增强（v1.6.0）==========
    
    def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        列出下载任务
        
        Args:
            status: 任务状态筛选（可选）：pending/processing/completed/failed
            limit: 返回数量限制
        
        Returns:
            任务列表
        
        Example:
            ```python
            # 列出所有任务
            tasks = client.list_tasks()
            
            # 只列出进行中的任务
            tasks = client.list_tasks(status="processing")
            ```
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        
        response = self._request("GET", "/api/v1/download/tasks", params=params)
        result = response.json()
        return result.get("tasks", [])
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消下载任务
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否成功
        
        Example:
            ```python
            success = client.cancel_task("task_id_here")
            if success:
                print("任务已取消")
            ```
        """
        try:
            response = self._request("DELETE", f"/api/v1/download/task/{task_id}")
            result = response.json()
            return result.get("code") == 200
        except Exception:
            return False
    
    def close(self):
        """关闭HTTP会话"""
        self.session.close()
    
    # ========== 内部辅助方法：参数验证 ==========
    
    def _validate_message_type(self, message_type: str):
        """验证消息类型格式"""
        if not message_type:
            raise ValidationError("消息类型不能为空")
        
        # 消息类型格式：ZZ-01, ZZ-31, ZZ-61 等
        if not message_type.startswith("ZZ-"):
            raise ValidationError(f"消息类型格式错误: {message_type}，应该是 ZZ-XX 格式（如 ZZ-01）")
        
        # 验证ZZ-后面是数字（不限制范围，因为会新增）
        suffix = message_type[3:]
        if not suffix or not suffix.isdigit():
            raise ValidationError(f"消息类型格式错误: {message_type}，ZZ-后面应该是数字")
    
    # symbol不验证 - 格式太灵活（SZ.000001, SZ.300978, SZ.163111等），交给后端处理
    
    def _validate_date(self, date: str):
        """验证日期格式"""
        if not date:
            raise ValidationError("日期不能为空")
        
        # 日期格式应该是 YYYYMMDD（8位数字）
        if not date.isdigit():
            raise ValidationError(f"日期格式错误: {date}，应该是8位数字（如 20251014）")
        
        if len(date) != 8:
            raise ValidationError(f"日期长度错误: {date}，应该是8位（如 20251014）")
        
        # 简单校验年月日的合法性
        try:
            year = int(date[0:4])
            month = int(date[4:6])
            day = int(date[6:8])
            
            if year < 2000 or year > 2100:
                raise ValidationError(f"年份超出范围: {year}")
            if month < 1 or month > 12:
                raise ValidationError(f"月份超出范围: {month}")
            if day < 1 or day > 31:
                raise ValidationError(f"日期超出范围: {day}")
        except ValueError as e:
            raise ValidationError(f"日期格式错误: {date}")
    
    def _validate_time(self, time_str: str):
        """验证时间格式"""
        if not time_str:
            raise ValidationError("时间不能为空")
        
        # 时间格式应该是 HHMM（4位数字）
        if not time_str.isdigit():
            raise ValidationError(f"时间格式错误: {time_str}，应该是4位数字（如 0930）")
        
        if len(time_str) != 4:
            raise ValidationError(f"时间长度错误: {time_str}，应该是4位（如 0930）")
        
        # 校验小时和分钟的合法性
        try:
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            
            if hour < 0 or hour > 23:
                raise ValidationError(f"小时超出范围: {hour}")
            if minute < 0 or minute > 59:
                raise ValidationError(f"分钟超出范围: {minute}")
        except ValueError:
            raise ValidationError(f"时间格式错误: {time_str}")
