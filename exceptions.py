# exceptions.py
"""
统一异常层级 — P2-4

所有自定义异常继承自 FinancialQABaseError，
便于上层统一捕获和日志记录，单题失败不中断全量测试。
"""


class FinancialQABaseError(Exception):
    """金融QA系统基础异常"""
    def __init__(self, message: str = "", details: dict = None):
        super().__init__(message)
        self.details = details or {}


# ---- 文档相关 ----
class PDFReadError(FinancialQABaseError):
    """PDF读取/解析失败"""
    pass


class DocumentNotFoundError(FinancialQABaseError):
    """文档未找到"""
    pass


class DocumentTooLargeError(FinancialQABaseError):
    """文档过大（超出页数或内存限制）"""
    pass


# ---- 检索相关 ----
class IndexBuildError(FinancialQABaseError):
    """索引构建失败"""
    pass


class RetrievalError(FinancialQABaseError):
    """检索执行失败"""
    pass


# ---- API相关 ----
class APIError(FinancialQABaseError):
    """Qwen API调用失败"""
    pass


class APITimeoutError(APIError):
    """API超时"""
    pass


class APIRateLimitError(APIError):
    """API限流"""
    pass


class APIQuotaError(APIError):
    """API额度耗尽"""
    pass


# ---- 答案相关 ----
class AnswerParseError(FinancialQABaseError):
    """答案提取/解析失败"""
    pass


class AnswerValidationError(FinancialQABaseError):
    """答案格式校验失败"""
    pass


# ---- 配置相关 ----
class ConfigError(FinancialQABaseError):
    """配置错误（如API Key未设置）"""
    pass


# ================================================================
# 错误处理辅助
# ================================================================
def safe_execute(func, *args, error_class=FinancialQABaseError, **kwargs):
    """
    安全执行函数，异常转为自定义异常并附加上下文。

    用法:
        result = safe_execute(parse_pdf, path, error_class=PDFReadError)
    """
    try:
        return func(*args, **kwargs)
    except FinancialQABaseError:
        raise
    except Exception as e:
        raise error_class(str(e), details={'original_error': str(e)}) from e
