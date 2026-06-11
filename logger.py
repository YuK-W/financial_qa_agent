# logger.py
"""
集中式日志模块 — P2-1

使用 loguru（优先）或标准 logging 作为后备。
所有模块通过 `from logger import log` 统一日志输出，
替代散落的 print() 调用。

特性:
  - 同时输出到控制台 + 文件
  - 文件按日期轮转
  - 异常自动附加 traceback
  - 兼容 Windows GBK 终端
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from loguru import logger as _loguru_logger

    # 移除默认 handler
    _loguru_logger.remove()

    # 日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 文件输出：DEBUG级别，按天轮转，保留7天
    _loguru_logger.add(
        os.path.join(log_dir, "agent_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        encoding="utf-8",
        format="{time:HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} | {message}",
    )

    # 控制台输出：INFO级别，去除emoji防GBK乱码
    _loguru_logger.add(
        sys.stderr,
        level="INFO",
        format="<level>{level: <7}</level> | {message}",
        colorize=False,
    )

    log = _loguru_logger

    HAS_LOGURU = True

except ImportError:
    # 后备：标准 logging
    import logging

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    _logger = logging.getLogger("financial_qa")
    _logger.setLevel(logging.DEBUG)

    # 文件 handler
    from logging.handlers import TimedRotatingFileHandler
    fh = TimedRotatingFileHandler(
        os.path.join(log_dir, "agent.log"),
        when="D", interval=1, backupCount=7, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    ))
    _logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    _logger.addHandler(ch)

    log = _logger

    HAS_LOGURU = False


def get_logger():
    """获取全局 logger 实例"""
    return log
