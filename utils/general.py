import os
from config.logger_config import logger_config
import logging.config

def create_dir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)

def init_log(name):
    config = logger_config
    config['handlers']['console']['level'] = 'INFO'
    logging.config.dictConfig(config)
    logger = logging.getLogger(name)
    return logger
