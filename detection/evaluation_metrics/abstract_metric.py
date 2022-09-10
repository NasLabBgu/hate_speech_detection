import logging
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from detection.detection_utils.factory import create_dir_if_missing

logger = logging.getLogger(__name__)


class AbstractMetric(ABC):
    name = None
    labels = None

    def __init__(self, **kwargs):
        self.df = None
        self.labels = kwargs["labels"]
        self.labels_mapping = kwargs["labels_mapping"]
        self.output_path = kwargs["output_path"]
    @abstractmethod
    def calc(self, y_true=None, y_true_dummy=None, y_pred=None, y_score=None, X_test=None):
        """
        :param y_test: True values
        :param y_pred: Predicted values
        :param y_score: Probability of prediction
        :param X_test: True features
        :return: None, only saves a Pandas.DataFrame with necessary data in self.df
        """
        logger.info(f'computing metric {self.name}')

    def save_plot(self, fig):
        create_dir_if_missing(self.output_path)
        file_path = os.path.join(self.output_path, f"{self.name}_plot.png")
        logger.info(f'saving plot metric {self.name} to {file_path}')
        fig.savefig(fname=file_path)
        plt.close(fig)
