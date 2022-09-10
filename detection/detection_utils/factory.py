import importlib
import logging
import os
logger = logging.getLogger(__name__)

def create_dir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)

def factory(module_class_string, super_cls: type = None, **kwargs):
    """
    :param module_class_string: full name of the class to create an object of
    :param super_cls: expected super class for validity, None if bypass
    :param kwargs: parameters to pass
    :return:
    """
    if "detection" not in module_class_string:
        module_class_string = f"detection.{module_class_string}"
    module_name, class_name = module_class_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, class_name), "class {} is not in {}".format(class_name, module_name)
    logger.debug('reading class {} from module {}'.format(class_name, module_name))
    cls = getattr(module, class_name)
    # if super_cls is not None:
    #     assert issubclass(cls, super_cls), "class {} should inherit from {}".format(class_name, super_cls.__name__)
    logger.debug('initialising {} with params {}'.format(class_name, kwargs))
    obj = cls(**kwargs)
    return obj
