import time
import logging

logger = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        logger.debug(f'starting {method.__module__}.{method.__name__}')
        result = method(*args, **kwargs)
        te = time.time()
        logger.info(f'finished {method.__module__}.{method.__name__} total {round(te - ts, 2):2.2f} sec elapsed')
        return result

    return timed
