import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import argparse


plt.clf()

y_true = np.load('result/' + "BL&GE_BERT" + '_true.npy')
y_scores = np.load('result/' + "BL&GE_BERT" + '_scores.npy')
precisions,recalls,threshold = precision_recall_curve(y_true, y_scores)
plt.plot(recalls, precisions, "-b", marker="+", markevery=200, lw=1, label="BL&GE_BERT")

plt.ylim([0.4, 1.0])
plt.xlim([0.0, 0.5])
plt.legend(loc="upper right")
plt.title("model performance")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()


