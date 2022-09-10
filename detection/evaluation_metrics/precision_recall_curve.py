import logging
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import seaborn as sns
sns.set(rc={'figure.figsize': (10, 10)}, font_scale=1.5)

from detection.evaluation_metrics.abstract_metric import AbstractMetric

logger = logging.getLogger(__name__)

class PrecisionRecallCurve(AbstractMetric):
    name = 'PrecisionRecallCurve'

    def calc(self, y_true=None, y_true_dummy=None, y_pred=None, y_score=None, X_test=None):
        super().calc()

        if len(self.labels) == 2:
            precisions, recalls, th = precision_recall_curve(y_true=y_true, probas_pred=y_score[:, -1])
            auc_ = auc(recalls, precisions)
            logger.info(f'Precision Recall curve AUC: {auc_}')
            fig = plt.figure()
            sns.lineplot(recalls, precisions, label=f'AUC = {auc_}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.axis('square')
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()
        else:  # multi class
            fig = plt.figure()
            for i in range(len(self.labels)):
                y_true_dummy = np.array(y_true_dummy)
                precisions, recalls, th = precision_recall_curve(y_true=y_true_dummy[:, i], probas_pred=y_score[:, i])
                plt.plot(recalls, precisions, lw=3, label=f'{self.labels[i]} (area={auc(recalls, precisions):0.2f})')
            plt.gca().set(xlim=[0.0,1.05], ylim=[0.0,1.05],
                      xlabel='Recall',
                      ylabel='Precision',
                      title="Precision-Recall curve")
            plt.legend(loc="best")
            plt.grid(True)
        super().save_plot(fig)
