
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

def metric_accuracy(preds, targets):
    pred_inds = np.argmax(np.asarray(preds), axis=1)
    return accuracy_score(targets, pred_inds)

def get_avg_f1(preds, labels):
    '''
    Calculate one-vs-all f1 score per class
    Return average of f1 scores over all classes
    preds: [num_samples x num_classes]
    '''
    preds = preds.tolist()
    labels = labels.tolist()
    f1s = []
    label_inds = labels
    class_inds_to_check = list(set(label_inds))

    for class_ind_to_check in class_inds_to_check:
        y_true = []
        y_pred = []
        for pred, lab_ind in zip(preds, label_inds):
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



# Load in predictions and targets

dataset1 = 'eval_in/'
dataset2 = 'eval_out/'

targets = np.concatenate((np.load(dataset1+'targets.npy'),np.load(dataset2+'targets.npy')))

preds1 = np.concatenate((np.load(dataset1+'1.npy'),np.load(dataset2+'1.npy')))
preds2 = np.concatenate((np.load(dataset1+'2.npy'),np.load(dataset2+'2.npy')))
preds3 = np.concatenate((np.load(dataset1+'3.npy'),np.load(dataset2+'3.npy')))
preds4 = np.concatenate((np.load(dataset1+'4.npy'),np.load(dataset2+'4.npy')))
preds5 = np.concatenate((np.load(dataset1+'5.npy'),np.load(dataset2+'5.npy')))
preds6 = np.concatenate((np.load(dataset1+'6.npy'),np.load(dataset2+'6.npy')))
preds7 = np.concatenate((np.load(dataset1+'7.npy'),np.load(dataset2+'7.npy')))
preds8 = np.concatenate((np.load(dataset1+'8.npy'),np.load(dataset2+'8.npy')))
preds9 = np.concatenate((np.load(dataset1+'9.npy'),np.load(dataset2+'9.npy')))
preds10 = np.concatenate((np.load(dataset1+'10.npy'),np.load(dataset2+'10.npy')))

preds = (preds1+preds2+preds3+preds4+preds5+preds6+preds7+preds8+preds9+preds10)/10
#preds = (preds1+preds2)/2

# Ensemble numbers
acc = metric_accuracy(preds, targets)
f1 = get_avg_f1(preds, targets)
print('Ens')
print('Accuracy:', acc)
print('Macro F1:', f1)

# Single seed results
preds_all = [preds1, preds2, preds3, preds4, preds5, preds6, preds7, preds8, preds9, preds10]
accs = []
f1s = []
for preds in preds_all:
    accs.append(metric_accuracy(preds,targets))
    f1s.append(get_avg_f1(preds,targets))
accs = np.asarray(accs)
f1s = np.asarray(f1s)
print('Single')
print('Accuracy:', accs.mean(), accs.std())
print('Macro F1:', f1s.mean(), f1s.std())

