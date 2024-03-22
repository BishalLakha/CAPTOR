import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score


def get_treshold(scores,alpha):
    return np.quantile(scores, alpha)

def labelAnomaly(scores, threshold):
    return scores<=threshold


def calculate_offline_performance_metrics(edge_scores, labels):
    print("\n\nCalculating Performance Metrics....")
    true_label = labels
    scores = edge_scores
    fpr, tpr, thresholds = metrics.roc_curve(true_label, scores, pos_label=1)
    fw = 0.5
    tw = 1 - fw
    fn = np.abs(tw * tpr - fw * (1 - fpr))
    best = np.argmin(fn, 0)
    print("\n\nOptimal cutoff %0.10f achieves TPR: %0.5f FPR: %0.5f on train data"
          % (thresholds[best], tpr[best], fpr[best]))
    auc = metrics.auc(fpr, tpr)
    print("Final AUC: ", auc)
    pred_label = np.where(scores >= thresholds[best], True, False)

    print("\n\n======= CLASSIFICATION REPORT =========\n")
    print(classification_report(true_label, pred_label))
    tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[False, True]).ravel()
    cm = confusion_matrix(true_label, pred_label, labels=[False, True])
    print("Confusion Matrix: \n", cm)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    print("FPR: ", fpr)
    print("TPR: ", tpr)
    
    return {'AUC': auc, 'FPR': fpr, 'TPR': tpr, 'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn}


def calculate_realtime_performance(train_score,test_score,percentile,labels):
    
    threshold = get_treshold(train_score,percentile)
    y_pred=labelAnomaly(-test_score, threshold)
    
    print('Area Under the ROC')
    print(roc_auc_score(labels,y_pred)) 
       
    print("\n\n======= CLASSIFICATION REPORT =========\n")
    print(classification_report(labels, y_pred))
    
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[False, True]).ravel()
    cm = confusion_matrix(labels, y_pred, labels=[False, True])
    print("Confusion Matrix: \n", cm)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    print("FPR: ", fpr)
    print("TPR: ", tpr)
    
    return {'FPR': fpr, 'TPR': tpr, 'true_positive': tp, 'false_positive': fp, 'true_negative': tn, 'false_negative': fn}

