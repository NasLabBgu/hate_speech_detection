import logging
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import roc_curve, auc
import numpy as np
import seaborn as sns
sns.set(rc={'figure.figsize': (10, 10)}, font_scale=1.5)
from detection.evaluation_metrics.abstract_metric import AbstractMetric

logger = logging.getLogger(__name__)

class ROC(AbstractMetric):
    name = 'ROC'

    def calc(self, y_true=None, y_true_dummy=None, y_pred=None, y_score=None, X_test=None):
        super().calc()

        if len(self.labels) == 2:
            fp, tp, th = roc_curve(y_true=y_true, y_score=y_score[:, -1])
            auc_ = auc(fp, tp)
            logger.info(f'AUC: {auc_}')
            fig = plt.figure()
            sns.lineplot(fp, tp, label=f'AUC = {auc_}')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC curve')
            plt.plot([0, 1], [0, 1], 'k:', label='random')
            plt.axis('square')
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()
        else:  # multi class
            fig = plt.figure()
            for i in range(len(self.labels)):
                # print(y_true_dummy)
                # print("COLUMNS: ", y_true_dummy.columns)
                # print(y_true_dummy.loc[:, float(i)])
                y_true_dummy = np.array(y_true_dummy)
                # print(y_true_dummy)
                # print(y_true_dummy[:, i])
                fp, tp, th = roc_curve(y_true=y_true_dummy[:, i], y_score=y_score[:, i])

                sns.lineplot(fp, tp, lw=3, label=f'{self.labels[i]} (area={auc(fp, tp):0.2f})')
            sns.lineplot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
            plt.gca().set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
                      xlabel='False Positive Rate',
                      ylabel="True Positive Rate (Recall)",
                      title="ROC curve")
            plt.legend(loc="best")
            plt.grid(True)
        super().save_plot(fig)
