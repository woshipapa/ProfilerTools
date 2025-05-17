import logging
import os
from datetime import datetime
import torch
from megatron.core import mpu
from megatron.core.models.hunyuan.model import HunyuanParams

logger = None

def get_logger():
    global logger
    if logger is None:
        logger = set_logger()
        return logger
    else:
        return logger


def set_logger():
    logging.getLogger().handlers.clear()

    tp_size = mpu.get_tensor_model_parallel_world_size()
    dp_size = mpu.get_data_parallel_world_size()
    num_layers = HunyuanParams().num_layers
    # 全局 logger 实例
    LOG_DIR = "hook_logs"
    LOGGER_NAME = "hook"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_file = datetime.now().strftime(f"{LOG_DIR}/logfile_%Y%m%d-%H%M%S_tp{tp_size}dp{dp_size}_num_layers{num_layers}.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    return logger