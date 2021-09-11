import numpy as np
import os

def classification_calibration(labels, probs, save_path, bins=10):
    n_classes = np.float(probs.shape[-1])
    lower = 0
    preds = np.argmax(probs, axis=1)
    total = labels.shape[0]
    probs = np.max(probs, axis=1)
    increment = 1.0 / bins
    upper = increment
    accs = np.zeros([bins + 1], dtype=np.float32)
    gaps = np.zeros([bins + 1], dtype=np.float32)
    confs = np.arange(0, 1.01, increment)
    ECE = 0.0
    for i in range(bins):
        ind1 = probs >= lower
        ind2 = probs < upper
        ind = np.where(np.logical_and(ind1, ind2))[0]
        lprobs = probs[ind]
        lpreds = preds[ind]
        llabels = labels[ind]
        acc = np.mean(np.asarray(llabels == lpreds, dtype=np.float32))
        prob = np.mean(lprobs)
        if np.isnan(acc):
            acc = 0.0
            prob = 0.0
        ECE += np.abs(acc - prob) * float(lprobs.shape[0])
        gaps[i] = np.abs(acc - prob)
        accs[i] = acc
        upper += increment
        lower += increment
    ECE /= np.float(total)
    MCE = np.max(np.abs(gaps))
    accs[-1] = 1.0
    fig, ax = plt.subplots(dpi=300)
    plt.plot(confs, accs)
    plt.plot(confs, confs)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.xlim(1.0/n_classes, 1.0)
    plt.legend(['Model','Ideal'])
    plt.savefig(os.path.join(save_path), bbox_inches='tight')
    plt.close()
    return np.round(ECE * 100.0, 2), np.round(MCE * 100.0, 2)


def eval_calibration(labels, probs, save_path, bins=10):
    likelihoods = probs[labels]
    nll = np.mean(-np.log(likelihoods))
    brier = np.mean((1.0-likelihoods)**2)
    ece, mce = classification_calibration(labels,probs,save_path=save_path, bins=bins)
    return np.round(nll,2), np.round(brier,2), np.round(ece,2), np.round(mce,2)


def negative_log_likelihood(temperature, labels, logits):
    probs = softmax(logits/math.exp(temperature), axis=1)
    likelihoods = probs[labels]
    nll = np.mean(-np.log(likelihoods))
    return nll


def negative_log_likelihood_ensembe(temperature, labels, logits):
    probs = np.mean(softmax(logits/math.exp(temperature), axis=2),axis=0)
    likelihoods = probs[labels]
    nll = np.mean(-np.log(likelihoods))
    return nll
