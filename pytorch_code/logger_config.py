import logging
import os


def setup_logger(log_filename):
    logger = logging.getLogger('my_logger')
    # 判断log_filename文件是否存在，如果不存在则创建
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            pass
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免日志输出重复
    # 如果handlers为空，则添加handler
    if not logger.handlers:
        handler = logging.FileHandler(log_filename)
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
