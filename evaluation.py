import numpy as np
from sklearn.metrics import average_precision_score
from utilities import prettyfloat

def evaluate_mAP(labels, predictions, testingFlag = False):

    def mAP(scores, labels):
        # unspecified = np.where(labels==0)[0]
        # scores = np.delete(scores, unspecified)
        # labels = np.delete(labels, unspecified)

        num_truths = sum(labels == 1)
        sort_ids = scores.argsort()[::-1]
        fp = np.cumsum(labels[sort_ids] == -1)
        tp = np.cumsum(labels[sort_ids] == 1)
        rec = tp / float(num_truths)
        prec = np.true_divide(tp, fp + tp)

        mrec = np.concatenate((np.array([0]), rec, np.array([1])), axis=0)
        mpre = np.concatenate((np.array([0]), prec, np.array([0])), axis=0)
        for i in range(mpre.shape[0] - 2, 0, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
        ap = sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap

    APs = []
    for att in range(np.array(predictions).shape[1]):
        if testingFlag:
            APs.append(mAP(predictions[:, att], labels[:, att]))
        else:
            APs.append(average_precision_score(labels[:, att], predictions[:, att]))
    mAP = sum(APs) / float(len(APs))
    return mAP, APs


def results(labels_tr, predicts_tr, labels_val, predicts_val, epoch, moving_loss_tr, moving_loss_val, elapsed_time):
    predicts_tr, labels_tr = np.array(predicts_tr), np.array(labels_tr)
    predicts_val, labels_val = np.array(predicts_val), np.array(labels_val)

    train_mAP, train_APs = evaluate_mAP(labels_tr, predicts_tr)
    val_mAP, val_APs = evaluate_mAP(labels_val, predicts_val)

    print("Epoch [%d]: Train-Loss=%f" % (epoch, moving_loss_tr))
    print("Epoch [%d]: Val-Loss=%f" % (epoch, moving_loss_val))
    print("Epoch [%d]: Train-mAP=%f" % (epoch, train_mAP))
    print("Epoch [%d]: Val-mAP=%f" % (epoch, val_mAP))
    print("Epoch [%d]: Elapsed-time=%f" % (epoch, elapsed_time))
    print(map(prettyfloat, train_APs))
    print(map(prettyfloat, val_APs))

    return train_mAP, train_APs, val_mAP, val_APs