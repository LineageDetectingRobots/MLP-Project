import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_curve(true_labels, probs):
    fpr, tpr, thresh = roc_curve(true_labels, probs)
    auc = roc_auc_score(true_labels, probs)
    print('AUC: %.3f' % auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="no skill")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    label = "ROC curve (area = %0.2f)" % auc
    plt.plot(fpr, tpr, color='darkred', lw=2, label=label)
    plt.title("ROC curve for baseline model")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import os, pandas as pd
    from framework import RESULTS_PATH
    results_csv_path = os.path.join(RESULTS_PATH, "results_baseline.csv")
    results_csv = pd.read_csv(results_csv_path)
    
    true_labels = results_csv.labels.values
    probs = results_csv.probs.values
    plot_roc_curve(true_labels, probs)