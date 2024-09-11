import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import precision_score
from scipy import signal
from scipy.stats import norm

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def get_events(y_test, outlier=1, normal=0):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events


def get_composite_fscore_raw(pred_labels, true_events, y_test, return_prec_rec=False):
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    prec_t = precision_score(y_test, pred_labels)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

def get_dynamic_scores(error_tc_train, error_tc_test, error_t_train, error_t_test, long_window=2000, short_window=10):
    # if error_tc is available, it will be used rather than error_t
    if error_tc_test is None:
        score_tc_test_dyn = None
        score_tc_train_dyn = None
        score_t_test_dyn = get_dynamic_score_t(error_t_train, error_t_test, long_window, short_window)
        if error_t_train is not None:
            score_t_train_dyn = get_dynamic_score_t(None, error_t_train, long_window, short_window)
        else:
            score_t_train_dyn = None
    else:
        n_cols = error_tc_test.shape[1]
        if error_tc_train is not None:
            score_tc_test_dyn = np.stack([get_dynamic_score_t(error_tc_train[:, col], error_tc_test[:, col],
                                                              long_window, short_window) for col in range(n_cols)],
                                         axis=-1)
            score_tc_train_dyn = np.stack([get_dynamic_score_t(None, error_tc_train[:, col],
                                                                    long_window, short_window) for col in range(n_cols)]
                                          , axis=-1)
            score_t_train_dyn = np.sum(score_tc_train_dyn, axis=1)
        else:
            score_tc_test_dyn = np.stack([get_dynamic_score_t(None, error_tc_test[:, col],
                                                              long_window, short_window) for col in range(n_cols)],
                                         axis=-1)
            score_t_train_dyn = None
            score_tc_train_dyn = None

        score_t_test_dyn = np.sum(score_tc_test_dyn, axis=1)

    return score_t_test_dyn, score_tc_test_dyn, score_t_train_dyn, score_tc_train_dyn
constant_std = 0.000001

def get_dynamic_score_t(error_t_train, error_t_test, long_window, short_window):
    n_t = error_t_test.shape[0]

    # assuming that length of scores is always greater than short_window
    short_term_means = np.concatenate((error_t_test[:short_window - 1], moving_average(error_t_test, short_window)))
    if long_window >= n_t:
        long_win = n_t - 1
    else:
        long_win = long_window

    if error_t_train is None:
        init_score_t_test = np.zeros(long_win - 1)
        means_test_t = moving_average(error_t_test, long_win)
        stds_test_t = np.array(pd.Series(error_t_test).rolling(window=long_win).std().values)[long_win - 1:]
        stds_test_t[stds_test_t == 0] = constant_std
        distribution = norm(0, 1)
        score_t_test_dyn = -distribution.logsf((short_term_means[(long_win - 1):] - means_test_t) / stds_test_t)
        score_t_test_dyn = np.concatenate([init_score_t_test, score_t_test_dyn])
    else:
        if len(error_t_train) < long_win - 1:
            full_ts = np.concatenate([np.zeros(long_win - 1 - len(error_t_train)), error_t_train, error_t_test], axis=0)
        else:
            full_ts = np.concatenate([error_t_train[-long_win + 1:], error_t_test], axis=0)
        means_test_t = moving_average(full_ts, long_win)
        stds_test_t = np.array(pd.Series(full_ts).rolling(window=long_win).std().values)[long_win - 1:]
        stds_test_t[stds_test_t == 0] = constant_std
        distribution = norm(0, 1)
        score_t_test_dyn = -distribution.logsf((short_term_means - means_test_t) / stds_test_t)

    return score_t_test_dyn

def moving_average(score_t, window=3):
    # return length = len(score_t) - window + 1
    ret = np.cumsum(score_t, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window

def get_gaussian_kernel_scores(score_t_dyn, score_tc_dyn, kernel_sigma):
    # if error_tc is available, it will be used rather than error_t
    gaussian_kernel = signal.gaussian(kernel_sigma * 8, std=kernel_sigma)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    if score_tc_dyn is None:
        score_tc_dyn_gauss_conv = None
        score_t_dyn_gauss_conv = signal.convolve(score_t_dyn, gaussian_kernel, mode="same")
    else:
        n_cols = score_tc_dyn.shape[1]
        score_tc_dyn_gauss_conv = np.stack([signal.convolve(score_tc_dyn[:, col], gaussian_kernel, mode="same")
                                      for col in range(n_cols)], axis=-1)
        score_t_dyn_gauss_conv = np.sum(score_tc_dyn_gauss_conv, axis=1)

    return score_t_dyn_gauss_conv, score_tc_dyn_gauss_conv

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    fbeta_score,
    average_precision_score,
)

default_thres_config = {
    "top_k_time": {},
    "best_f1_test": {"exact_pt_adj": True},
    "thresholded_score": {},
    "tail_prob": {"tail_prob": 2},
    "tail_prob_1": {"tail_prob": 1},
    "tail_prob_2": {"tail_prob": 2},
    "tail_prob_3": {"tail_prob": 3},
    "tail_prob_4": {"tail_prob": 4},
    "tail_prob_5": {"tail_prob": 5},
    "dyn_gauss": {"long_window": 10000, "short_window": 1, "kernel_sigma": 120},
    "nasa_npt": {
        "batch_size": 70,
        "window_size": 30,
        "telem_only": True,
        "smoothing_perc": 0.005,
        "l_s": 250,
        "error_buffer": 5,
        "p": 0.05,
    },
}


def get_f_score(prec, rec):
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec)
    return f_score


def get_point_adjust_scores(y_test, pred_labels, true_events):
    tp = 0
    fn = 0
    for true_event in true_events.keys():
        true_start, true_end = true_events[true_event]
        if pred_labels[true_start:true_end].sum() > 0:
            tp += true_end - true_start
        else:
            fn += true_end - true_start
    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)

    prec, rec, fscore = get_prec_rec_fscore(tp, fp, fn)
    return fp, fn, tp, prec, rec, fscore


def get_prec_rec_fscore(tp, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    fscore = get_f_score(precision, recall)
    return precision, recall, fscore


def get_composite_fscore_from_scores(
    score_t_test, thres, true_events, prec_t, return_prec_rec=False
):
    pred_labels = score_t_test > thres
    tp = np.sum(
        [pred_labels[start : end + 1].any() for start, end in true_events.values()]
    )
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


def threshold_and_predict(
    score_t_test,
    y_test,
    true_events,
    logger,
    test_anom_frac,
    thres_method="top_k_time",
    point_adjust=False,
    score_t_train=None,
    thres_config_dict=dict(),
    return_auc=False,
    composite_best_f1=False,
):
    if thres_method in thres_config_dict.keys():
        config = thres_config_dict[thres_method]
    else:
        config = default_thres_config[thres_method]
    # test_anom_frac = (np.sum(y_test)) / len(y_test)
    auroc = None
    avg_prec = None
    if thres_method == "thresholded_score":
        opt_thres = 0.5
        if set(score_t_test) - {0, 1}:
            logger.error("Score_t_test isn't binary. Predicting all as non-anomalous")
            pred_labels = np.zeros(len(score_t_test))
        else:
            pred_labels = score_t_test

    elif thres_method == "best_f1_test" and point_adjust:
        prec, rec, thresholds = precision_recall_curve(
            y_test, score_t_test, pos_label=1
        )
        if not config["exact_pt_adj"]:
            fscore_best_time = [
                get_f_score(precision, recall) for precision, recall in zip(prec, rec)
            ]
            opt_num = np.squeeze(np.argmax(fscore_best_time))
            opt_thres = thresholds[opt_num]
            thresholds = np.random.choice(thresholds, size=5000) + [opt_thres]
        fscores = []
        for thres in thresholds:
            _, _, _, _, _, fscore = get_point_adjust_scores(
                y_test, score_t_test > thres, true_events
            )
            fscores.append(fscore)
        opt_thres = thresholds[np.argmax(fscores)]
        pred_labels = score_t_test > opt_thres

    elif thres_method == "best_f1_test" and composite_best_f1:
        prec, rec, thresholds = precision_recall_curve(
            y_test, score_t_test, pos_label=1
        )
        precs_t = prec
        fscores_c = [
            get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t)
            for thres, prec_t in zip(thresholds, precs_t)
        ]
        try:
            opt_thres = thresholds[np.nanargmax(fscores_c)]
        except:
            opt_thres = 0.0
        pred_labels = score_t_test > opt_thres

    elif thres_method == "top_k_time":
        opt_thres = np.nanpercentile(
            score_t_test, 100 * (1 - test_anom_frac), interpolation="higher"
        )
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif thres_method == "best_f1_test":
        prec, rec, thres = precision_recall_curve(y_test, score_t_test, pos_label=1)
        fscore = [
            get_f_score(precision, recall) for precision, recall in zip(prec, rec)
        ]
        opt_num = np.squeeze(np.argmax(fscore))
        opt_thres = thres[opt_num]
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif "tail_prob" in thres_method:
        tail_neg_log_prob = config["tail_prob"]
        opt_thres = tail_neg_log_prob
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif thres_method == "nasa_npt":
        opt_thres = 0.5
        pred_labels = get_npt_labels(score_t_test, y_test, config)
    else:
        logger.error(
            "Thresholding method {} not in [top_k_time, best_f1_test, tail_prob]".format(
                thres_method
            )
        )
        return None, None
    if return_auc:
        avg_prec = average_precision_score(y_test, score_t_test)
        auroc = roc_auc_score(y_test, score_t_test)
        return opt_thres, pred_labels, avg_prec, auroc
    return opt_thres, pred_labels