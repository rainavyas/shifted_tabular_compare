import numpy as np
from sklearn.metrics import precision_recall_curve

def get_class_ind(preds):
    return np.argmax(np.asarray(preds), axis=1)

def get_accuracy(preds, labels):
    '''
    preds: [num_samples x num_classes]
    '''
    pred_class_inds = get_class_ind(preds)
    total = 0
    num_correct = 0
    for pred, lab in zip(pred_class_inds, labels):
        if pred == lab:
            num_correct+=1
        total+=1
    return num_correct/total

def get_avg_f1(preds, labels):
    '''
    Calculate one-vs-all f1 score per class
    Return average of f1 scores over all classes
    preds: [num_samples x num_classes]
    '''
    f1s = []
    class_inds_to_check = list(set(labels))

    for class_ind_to_check in class_inds_to_check:
        y_true = []
        y_pred = []
        for pred, lab_ind in zip(preds, labels):
            y_pred.append(pred[class_ind_to_check])
            if lab_ind == class_ind_to_check:
                y_true.append(1)
            else:
                y_true.append(0)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        f_scores = (2*precision*recall)/(precision+recall)
        f_scores_clean = f_scores[np.logical_not(np.isnan(f_scores))]
        f1s.append(np.amax(f_scores_clean))
    return np.mean(np.asarray(f1s))