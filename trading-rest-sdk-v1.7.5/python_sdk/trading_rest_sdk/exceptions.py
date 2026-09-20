"""
REST SDK异常定义
"""


class TradingRestError(Exception):
    """SDK基础异常类"""
    pass


class APIError(TradingRestError):
    """API请求错误"""
    
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(TradingRestError):
    """认证错误"""
    pass


class NotFoundError(TradingRestError):
    """资源不存在"""
    pass


class TaskError(TradingRestError):
    """下载任务错误"""
    pass


class ValidationError(TradingRestError):
    """参数验证错误"""
    pass
