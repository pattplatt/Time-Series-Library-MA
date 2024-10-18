import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import signal
from scipy.stats import norm
from datetime import datetime
import csv
from numba import njit
import pickle
from concurrent.futures import ProcessPoolExecutor
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
    confusion_matrix,
    precision_recall_fscore_support
)
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

def fit_univar_distr(scores_arr: np.ndarray, distr='univar_gaussian'):
    """
    :param scores_arr: 1d array of reconstruction errors
    :param distr: the name of the distribution to be fitted to anomaly scores on train data
    :return: params dict with distr name and parameters of distribution
    """
    distr_params = {'distr': distr}
    constant_std = 0.000001
    if distr == "univar_gaussian":
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        if std == 0.0:
            std += constant_std
        distr_params["mean"] = mean
        distr_params["std"] = std
    elif distr == "univar_lognormal":
        shape, loc, scale = lognorm.fit(scores_arr)
        distr_params["shape"] = shape
        distr_params["loc"] = loc
        distr_params["scale"] = scale
    elif distr == "univar_lognorm_add1_loc0":
        shape, loc, scale = lognorm.fit(scores_arr + 1.0, floc=0.0)
        if shape == 0.0:
            shape += constant_std
        distr_params["shape"] = shape
        distr_params["loc"] = loc
        distr_params["scale"] = scale
    elif distr == "chi":
        estimated_df = chi.fit(scores_arr)[0]
        df = round(estimated_df)
        distr_params["df"] = df
    else:
        print("This distribution is unknown or has not been implemented yet, a univariate gaussian will be used")
        mean = np.mean(scores_arr)
        std = np.std(scores_arr)
        distr_params["mean"] = mean
        distr_params["std"] = std
    return distr_params

def get_scores_channelwise(distr_params, train_raw_scores, val_raw_scores, test_raw_scores, drop_set=set([]), logcdf=False):
    use_ch = list(set(range(test_raw_scores.shape[1])) - drop_set)
    train_prob_scores = -1 * np.concatenate(([get_per_channel_probas(train_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
                                        for i in range(train_raw_scores.shape[1])]), axis=1)
    test_prob_scores = [get_per_channel_probas(test_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
                                       for i in range(train_raw_scores.shape[1])]
    test_prob_scores = -1 * np.concatenate(test_prob_scores, axis=1)

    train_ano_scores = np.sum(train_prob_scores[:, use_ch], axis=1)
    test_ano_scores = np.sum(test_prob_scores[:, use_ch], axis=1)

    if val_raw_scores is not None:
        val_prob_scores = -1 * np.concatenate(([get_per_channel_probas(val_raw_scores[:, i].reshape(-1, 1), distr_params[i], logcdf=logcdf)
                                          for i in range(train_raw_scores.shape[1])]), axis=1)
        val_ano_scores = np.sum(val_prob_scores[:, use_ch], axis=1)
    else:
        val_ano_scores = None
        val_prob_scores = None
    return train_ano_scores, val_ano_scores, test_ano_scores, train_prob_scores, val_prob_scores, test_prob_scores

# Computes (when not already saved) parameters for scoring distributions
def fit_distributions(distr_par_file, distr_names, predictions_dic, val_only=False):
    #try:
        #with open(distr_par_file, 'rb') as file:
            #distributions_dic = pickle.load(file)
    #except:
        #print("not found")
    distributions_dic = {}
    for distr_name in distr_names:
        if distr_name in distributions_dic.keys():
            continue
        else:
            print("The distribution parameters for %s for this algorithm on this data set weren't found. \
            Will fit them" % distr_name)
            if "val_raw_scores" in predictions_dic:
                raw_scores = np.concatenate((predictions_dic["train_raw_scores"], predictions_dic["val_raw_scores"]))
            else:
                raw_scores = predictions_dic["train_raw_scores"]
            if raw_scores.ndim == 1:
                raw_scores = raw_scores.reshape(-1, 1)
                
            distributions_dic[distr_name] = [fit_univar_distr(raw_scores[:, i], distr=distr_name)
                                             for i in range(raw_scores.shape[1])]
    with open(distr_par_file, 'wb') as file:
        pickle.dump(distributions_dic, file)

    return distributions_dic

def get_per_channel_probas(pred_scores_arr, params, logcdf=False):
    """
    :param pred_scores_arr: 1d array of the reconstruction errors for one channel
    :param params: must contain key 'distr' and corresponding params
    :return: array of negative log pdf of same length as pred_scores_arr
    """
    distr = params["distr"]
    probas = None
    constant_std = 0.000001
    if distr == "univar_gaussian":
        assert ("mean" in params.keys() and ("std" in params.keys()) or "variance" in params.keys()), \
            "The mean and/or standard deviation are missing, we can't define the distribution"
        if "std" in params.keys():
            if params["std"] == 0.0:
                params["std"] += constant_std
            distribution = norm(params["mean"], params["std"])

        else:
            distribution = norm(params["mean"], np.sqrt(params["variance"]))
    elif distr == "univar_lognormal":
        assert ("shape" in params.keys() and "loc" in params.keys() and "scale" in params.keys()), "The shape or scale \
                    or loc are missing, we can't define the distribution"
        shape = params["shape"]
        loc = params["loc"]
        scale = params["scale"]
        distribution = lognorm(s=shape, loc=loc, scale=scale)
    elif distr == "univar_lognorm_add1_loc0":
        assert ("shape" in params.keys() and "loc" in params.keys() and "scale" in params.keys()), "The shape or scale \
                    or loc are missing, we can't define the distribution"
        shape = params["shape"]
        loc = params["loc"]
        scale = params["scale"]
        distribution = lognorm(s=shape, loc=loc, scale=scale)
        if logcdf:
            probas = distribution.logsf(pred_scores_arr + 1.0)
        else:
            probas = distribution.logpdf(pred_scores_arr + 1.0)
    elif distr == "chi":
        assert "df" in params.keys(), "The number of degrees of freedom is missing, we can't define the distribution"
        df = params["df"]
        distribution = chi(df)
    else:
        print("This distribution is unknown or has not been implemented yet, a univariate gaussian will be used")
        assert ("mean" in params.keys() and "std" in params.keys()), "The mean and/or standard deviation are missing, \
        we can't define the distribution"
        distribution = norm(params["mean"], params["std"])

    if probas is None:
        if logcdf:
            probas = distribution.logsf(pred_scores_arr)
        else:
            probas = distribution.logpdf(pred_scores_arr)

    return probas
    
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

def get_negative_intervals(true_events, total_length):
    negative_intervals = []
    prev_end = -1
    sorted_events = sorted(true_events.values())

    for start, end in sorted_events:
        if prev_end + 1 < start:
            # There is a gap between the previous event and the current event
            negative_intervals.append((prev_end + 1, start - 1))
        prev_end = end

    # Check for a negative interval after the last event
    if prev_end + 1 < total_length:
        negative_intervals.append((prev_end + 1, total_length - 1))

    return negative_intervals

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

default_thres_config = {
    "top_k_time": {},
    "best_f1_test": {"exact_pt_adj": False},
    "thresholded_score": {},
    "tail_prob": {"tail_prob": 2*123},
    "tail_prob_1": {"tail_prob": 123},
    "tail_prob_2": {"tail_prob": 2*123},
    "tail_prob_3": {"tail_prob": 3*123},
    "tail_prob_4": {"tail_prob": 4*123},
    "tail_prob_5": {"tail_prob": 5*123},
    "dyn_gauss": {"long_window": 100000, "short_window": 1, "kernel_sigma": 120},
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

def get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t, return_prec_rec=False):
    pred_labels = score_t_test > thres
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

@njit
def compute_tp(pred_labels, event_starts, event_ends):
    tp = 0
    for i in range(len(event_starts)):
        start = event_starts[i]
        end = event_ends[i]
        for j in range(start, end + 1):
            if pred_labels[j]:
                tp += 1
                break  # Stop after first positive in event
    return tp

def get_composite_fscore_from_scores_optimized(score_t_test, thres,true_events ,prec_t):
    
    event_starts = np.array([start for start, end in true_events.values()])
    event_ends = np.array([end for start, end in true_events.values()])
    
    pred_labels = score_t_test > thres
    pred_labels = pred_labels.astype(np.bool_)
    tp = compute_tp(pred_labels, event_starts, event_ends)
    fn = len(event_starts) - tp
    rec_e = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fscore_c = (
        2 * rec_e * prec_t / (rec_e + prec_t) if (rec_e + prec_t) > 0 else 0.0
    )
    return fscore_c

from functools import partial

def compute_fscore_c(args, score_t_test,true_events):
    thres, prec_t = args
    return get_composite_fscore_from_scores(score_t_test, thres, true_events,prec_t)

def threshold_and_predict(
    score_t_test,
    y_test,
    true_events,
    logger,
    test_anom_frac=0.01,
    thres_method="top_k_time",
    point_adjust=False,
    score_t_train=None,
    thres_config_dict=dict(),
    return_auc=False,
    composite_best_f1=False,
    score_t_test_and_train=None,
):
    if thres_method in thres_config_dict.keys():
        config = thres_config_dict[thres_method]
    else:
        config = default_thres_config[thres_method]
        
    if score_t_test_and_train is not None:        
        test_anom_frac = (np.sum(y_test)) / len(score_t_test_and_train)
    else:
        test_anom_frac = (np.sum(y_test)) / len(y_test)
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
        prec, rec, thresholds = precision_recall_curve(y_test, score_t_test)
        precs_t = prec        
        # Prepare args_list
        #args_list = list(zip(thresholds, precs_t))
        # Fix score_t_test using partial
        #compute_fscore_c_partial = partial(compute_fscore_c, score_t_test=score_t_test, true_events=true_events)
        #with ProcessPoolExecutor() as executor:
         #   fscores_c = list(executor.map(compute_fscore_c_partial, args_list))

        fscores_c = [get_composite_fscore_from_scores(score_t_test, thres, true_events,prec_t) for thres, prec_t in zip(thresholds, precs_t)]
        try:
            opt_thres = thresholds[np.nanargmax(fscores_c)]
        except:
            opt_thres = 0.0
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif thres_method == "top_k_time":
        if score_t_test_and_train is not None:
            opt_thres = np.nanpercentile(
                score_t_test_and_train, 100 * (1 - test_anom_frac), interpolation="higher"
            )
        else:
            opt_thres = np.nanpercentile(
                score_t_test, 100 * (1 - test_anom_frac), interpolation="higher"
            )
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)
            
    elif thres_method == "best_f1_test":
        prec, rec, thres = precision_recall_curve(y_test, score_t_test)
        
        fscore = [
            get_f_score(precision, recall) for precision, recall in zip(prec, rec)
        ]
        opt_num = np.squeeze(np.nanargmax(fscore))
        opt_thres = thres[opt_num]
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    elif "tail_prob" in thres_method:
        max_fscore = -1
        best_pred_labels = None
        best_tail_neg_log_prob = None
        best_key = None  # To store the key corresponding to the best f-score

        # Loop over all tail_prob values in the config
        for key in default_thres_config.keys():
            if key.startswith("tail_prob_"):

                tail_neg_log_prob = default_thres_config[key]["tail_prob"]
                opt_thres = tail_neg_log_prob

                pred_labels = np.where(score_t_test > opt_thres, 1, 0)

                precision, recall, fscore, _ = precision_recall_fscore_support(
                    y_test, pred_labels, average='binary'
                )
                # Keep the pred_labels with the highest f-score
                if fscore > max_fscore:
                    max_fscore = fscore
                    best_pred_labels = pred_labels
                    best_tail_neg_log_prob = tail_neg_log_prob
                    best_key = key  # Store the key without overwriting 'key'

        # After evaluating all tail_prob values, keep the best one
        pred_labels = best_pred_labels
        opt_thres = best_tail_neg_log_prob
        print("tail_prob")
        print(f"opt_thres:{opt_thres}")
        print(f"fscore:{fscore}")

    elif thres_method == "nasa_npt":
        opt_thres = 0.5
        pred_labels = get_npt_labels(score_t_test, y_test, config)
    else:
        print("Thresholding method {} not in [top_k_time, best_f1_test, tail_prob]")
        logger.error(
            "Thresholding method {} not in [top_k_time, best_f1_test, tail_prob]".format(
                thres_method
            )
        )
        return None, None
        
    if return_auc:
        avg_prec = average_precision_score(y_test, score_t_test)
        auroc = roc_auc_score(y_test, score_t_test)
        return opt_thres, pred_labels, avg_prec, auroc, test_anom_frac
    return opt_thres, pred_labels, test_anom_frac

def evaluate_metrics(gt, pred, auroc, pred_best_fpa=None, pred_best_fc=None, auroc_pa=None, auroc_c=None, seq_len=1):
    # Generals Accuracy and Confusion Matrix
    accuracy = accuracy_score(gt, pred)
    cm = confusion_matrix(gt, pred)
    #normal F1-Score
    if seq_len == 0:
        print("Warning: seq_len is zero. Normalized confusion matrix cannot be computed.")
        cm_normalized = cm
    else:
        cm_normalized = np.floor(cm / seq_len)

    precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')

    # Point-adjusted F1 Score
    try:
        if pred_best_fpa is not None:
            labels_point_adjusted, anomaly_preds_point_adjusted = adjustment(gt, pred_best_fpa.copy())
        else:
            labels_point_adjusted, anomaly_preds_point_adjusted = adjustment(gt, pred.copy())
            
        accuracy_point_adjusted = accuracy_score(labels_point_adjusted, anomaly_preds_point_adjusted)
        precision_point_adjusted, recall_point_adjusted, f_score_point_adjusted, _ = precision_recall_fscore_support(
        labels_point_adjusted, anomaly_preds_point_adjusted, average='binary')
        cm_point_adjusted = confusion_matrix(labels_point_adjusted, anomaly_preds_point_adjusted)
        if auroc_pa is not None:
            auroc_pa=auroc_pa
        else:
            auroc_pa=auroc
        
    except NameError:
        print("Error: 'adjustment' function is not defined. Skipping point-adjusted metrics.")
        accuracy_point_adjusted = precision_point_adjusted = recall_point_adjusted = f_score_point_adjusted = None
        cm_point_adjusted = None

    # Composite F1 Score
    try:
        true_events = get_events(gt)
        if pred_best_fc is not None:
            prec_t, rec_e, fscore_c= get_composite_fscore_raw(pred_best_fc.copy(), true_events, gt, return_prec_rec=True)
        else:               
            prec_t, rec_e, fscore_c= get_composite_fscore_raw(pred.copy(), true_events, gt, return_prec_rec=True)
        if auroc_c is not None:
            auroc_c=auroc_c
        else:
            auroc_c=auroc
        
    except NameError:
        prec_t = rec_e = fscore_c = None

    # Prepare metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f_score': f_score,
        'confusion_matrix': cm,
        'cm_normalized': cm_normalized,
        'accuracy_point_adjusted': accuracy_point_adjusted,
        'precision_point_adjusted': precision_point_adjusted,
        'recall_point_adjusted': recall_point_adjusted,
        'f_score_point_adjusted': f_score_point_adjusted,
        'cm_point_adjusted': cm_point_adjusted,
        'acc_c': accuracy,
        'prec_t': prec_t,
        'rec_e': rec_e,
        'fscore_c': fscore_c,
        'auroc':auroc,
        'auroc_pa':auroc_pa,
        'auroc_c':auroc_c,
    }
    return metrics

def write_to_csv(
    model_name, model_id,avg_train_loss, avg_vali_loss, avg_test_loss, seq_len, d_model,enc_in, e_layers,dec_in,d_layers,c_out,
    d_ff, n_heads, long_window, short_window, kernel_sigma, train_epochs, learning_rate, batch_size,anomaly_ratio, embed, total_time, train_duration,
    test_duration, metrics, test_results_path, setting, benchmark_id, total_allocated_memory_usage, total_reserved_memory_usage
):
    # Parameters header and data
    parameters_header = [
        "Model", "avg_train_loss", "avg_vali_loss", "avg_test_loss", "seq_len", "d_model", "enc_in","e_layers", "dec_in", "d_layers", "c_out","d_ff","n_heads","long_window","short_window","kernel_sigma","train_epochs","learning_rate","batch_size","anomaly_ratio", "embed", "Total Duration (min)", "Train Duration (min)", "Test Duration (min)", "Average allocated memory (MB)", "Average reserved memory (MB)"]
    parameters = [
        model_name, f"{avg_train_loss:.6f}", f"{avg_vali_loss:.6f}", f"{avg_test_loss:.6f}", seq_len, d_model, enc_in,e_layers, dec_in,d_layers,c_out,d_ff,n_heads, long_window, short_window, kernel_sigma, train_epochs, learning_rate, batch_size, anomaly_ratio, embed, f"{(total_time / 60):.2f}",f"{(train_duration/60):.2f}", f"{(test_duration/60):.2f}",  f"{total_allocated_memory_usage:.2f}",  f"{total_reserved_memory_usage:.2f}"]

    file_prefix = os.path.join(test_results_path, "results_"+ model_id )
    parameters_file = file_prefix + "_parameters.csv"
    metrics_file = file_prefix + "_metrics.csv"

    # Write parameters to CSV
    file_exists = os.path.isfile(parameters_file)
    with open(parameters_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(parameters_header)
        writer.writerow(parameters)

    # Metrics header
    metrics_header = ["Model", "Metric Type", "Accuracy", "Precision", "Recall", "F-1", "AUROC", "Confusion Matrix"]

    # Function to write metrics to CSV
    def write_metrics(writer, model_name, metrics_dict):
        
        # Default F1 metrics
        metrics_f1 = [
            model_name, "Default F1", f"{metrics_dict.get('accuracy', 'N/A'):.4f}",
            f"{metrics_dict.get('precision', 'N/A'):.4f}", f"{metrics_dict.get('recall', 'N/A'):.4f}",
            f"{metrics_dict.get('f_score', 'N/A'):.4f}" ,f"{metrics_dict.get('auroc', 'N/A'):.4f}", f"{metrics_dict.get('confusion_matrix', 'N/A')}"
        ]
        writer.writerow(metrics_f1)

        # Point-adjusted metrics
        if metrics_dict.get('accuracy_point_adjusted') is not None:
            metrics_fpa = [
                model_name, "Fpa", f"{metrics_dict['accuracy_point_adjusted']:.4f}",
                f"{metrics_dict['precision_point_adjusted']:.4f}", f"{metrics_dict['recall_point_adjusted']:.4f}",
                f"{metrics_dict['f_score_point_adjusted']:.4f}",f"{metrics_dict.get('auroc_pa', 'N/A'):.4f}", f"{metrics_dict['cm_point_adjusted']}"
            ]
        else:
            metrics_fpa = [model_name, "Fpa", "N/A", "N/A", "N/A", "N/A", "N/A"]
        writer.writerow(metrics_fpa)

        # Composite F1 metrics
        if metrics_dict.get('fscore_c') is not None:
            metrics_fc = [
                model_name, "Fc", f"{metrics_dict['acc_c']:.4f}", f"{metrics_dict['prec_t']:.4f}", f"{metrics_dict['rec_e']:.4f}", f"{metrics_dict['fscore_c']:.4f}",f"{metrics_dict.get('auroc_c', 'N/A'):.4f}", "-"
            ]
        else:
            metrics_fc = [model_name, "Fc", "N/A", "N/A", "N/A", "N/A", "-"]
        writer.writerow(metrics_fc)
    
    # Write metrics to CSV
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Check if file exists to write header
        file_exists = os.path.isfile(metrics_file)
        if not file_exists:
            writer.writerow(metrics_header)
        
        elif isinstance(metrics, dict) and len(metrics) == 3:
            
            writer.writerow(metrics_header)
            for key, metric in metrics.items():
            # Write the first metrics dict
                writer.writerow([])
                writer.writerow([key])
                for key, scoring in metric.items():
                    writer.writerow([])
                    writer.writerow([key])
                    write_metrics(writer, model_name + "_" + key, scoring)

        else:
            # Handle unexpected metrics format
            raise ValueError("The 'metrics' parameter should be a dict or a list containing two dicts.")
    
    write_f_score_metrics(benchmark_id, model_name, model_id, metrics, test_results_path)
            
def write_f_score_metrics(benchmark_id, model_name, model_id, metrics, test_results_path):

    # Get the parent directory of test_results_path
    parent_dir = os.path.dirname(test_results_path)

    # Construct the path for aggregate_metrics.csv in the parent directory
    metrics_csv_file = os.path.join(parent_dir, benchmark_id + "_aggregate_metrics.csv")
    file_exists = os.path.isfile(metrics_csv_file)

    # Metrics we are interested in
    metrics_of_interest = ['f_score', 'f_score_point_adjusted', 'fscore_c', 'auroc']

    # Prepare data to write: collect metrics into a dictionary
    data_to_write = {}

    # Collect metrics from the provided metrics dictionary
    # and compute per-threshold-method mean scores
    overall_fscores = []
    overall_aurocs = []

    for thres_method, method_metrics in metrics.items():
        method_fscores = []
        method_aurocs = []
        for score_type, metric_values in method_metrics.items():
            # Construct a unique key for each metric type
            metric_key_prefix = f"{thres_method}_{score_type}"
            for metric_name in metrics_of_interest:
                if metric_name in metric_values:
                    data_key = f"{metric_key_prefix}_{metric_name}"
                    value = metric_values[metric_name]
                    if isinstance(value, (int, float)):
                        # Format to 5 decimal points
                        value_formatted = f"{value:.5f}"
                        data_to_write[data_key] = value_formatted
                        # Collect values for mean computation
                        if metric_name == 'f_score':
                            method_fscores.append(float(value_formatted))
                            overall_fscores.append(float(value_formatted))
                        if metric_name == 'auroc':
                            method_aurocs.append(float(value_formatted))
                            overall_aurocs.append(float(value_formatted))
                    else:
                        data_to_write[data_key] = value  # For non-numeric values
        # Compute mean f_score and auroc per threshold method
        method_mean_fscore = sum(method_fscores) / len(method_fscores) if method_fscores else None
        method_mean_auroc = sum(method_aurocs) / len(method_aurocs) if method_aurocs else None
        # Format mean values to 5 decimal points
        if method_mean_fscore is not None:
            data_to_write[f"{thres_method}_mean_fscore"] = f"{method_mean_fscore:.5f}"
        else:
            data_to_write[f"{thres_method}_mean_fscore"] = "N/A"
        if method_mean_auroc is not None:
            data_to_write[f"{thres_method}_mean_auroc"] = f"{method_mean_auroc:.5f}"
        else:
            data_to_write[f"{thres_method}_mean_auroc"] = "N/A"

    # Compute overall mean fscore and auroc per model (over all methods and score types)
    overall_mean_fscore = sum(overall_fscores) / len(overall_fscores) if overall_fscores else None
    overall_mean_auroc = sum(overall_aurocs) / len(overall_aurocs) if overall_aurocs else None
    # Format overall mean values to 5 decimal points
    if overall_mean_fscore is not None:
        data_to_write['mean_fscore'] = f"{overall_mean_fscore:.5f}"
    else:
        data_to_write['mean_fscore'] = "N/A"
    if overall_mean_auroc is not None:
        data_to_write['mean_auroc'] = f"{overall_mean_auroc:.5f}"
    else:
        data_to_write['mean_auroc'] = "N/A"

    # Now, prepare a DataFrame with metrics as rows and models as columns
    # If the CSV file exists, read it
    if os.path.isfile(metrics_csv_file):
        df = pd.read_csv(metrics_csv_file, index_col=0)
    else:
        df = pd.DataFrame()

    # Convert data_to_write to a DataFrame
    model_df = pd.DataFrame.from_dict(data_to_write, orient='index', columns=[model_name+"_"+model_id])

    # Combine with existing DataFrame
    df = df.join(model_df, how='outer')

    # Save the DataFrame to CSV
    df.to_csv(metrics_csv_file)

def compute_metrics(test_energy, gt, true_events, score_t_test_dyn, score_t_test_dyn_gauss_conv, seq_len,train_energy=None,  score_t_train_dyn=None,score_t_train_dyn_gauss_conv=None):

    metrics = {}

    # Prepare score dictionaries
    scores = {
        'default': test_energy,
        'dyn': score_t_test_dyn,
        'dyn_gauss': score_t_test_dyn_gauss_conv,
    }

    combined_scores = {}
    
    if train_energy is not None:
        combined_scores['default'] = np.concatenate([test_energy, train_energy])
    else:
        combined_scores['default'] = None

    # Add 'dyn' scores if provided
    if score_t_train_dyn is not None:
        combined_scores['dyn'] = np.concatenate([score_t_train_dyn, score_t_test_dyn])
    else:
        combined_scores['dyn'] = None

    # Add 'dyn_gauss' scores if provided
    if score_t_train_dyn_gauss_conv is not None:
        combined_scores['dyn_gauss'] = np.concatenate([score_t_train_dyn_gauss_conv,score_t_test_dyn_gauss_conv])
    else:
        combined_scores['dyn_gauss'] = None

    # Thresholding methods to evaluate
    thres_methods = ['best_f1_test', 'top_k_time', 'tail_prob']
    metrics_methods = {}

    for thres_method in thres_methods:
        method_metrics = {}
        for score_type in scores.keys():
            score = scores[score_type]
            if score is None:
                continue  # Skip if score is not provided
            # Prepare additional arguments for threshold_and_predict
            thres_args = {}
            if thres_method == 'top_k_time' or thres_method == 'best_f1_test':
                thres_args['score_t_test_and_train'] = None
            # If combined_scores[score_type] is None, threshold_and_predict should handle it
            if thres_method == 'best_f1_test':
                # Apply threshold and predict
                thres, pred_labels, avg_prec, auroc ,test_anom_frac= threshold_and_predict(
                    score, gt, true_events=true_events, logger=None,
                    thres_method=thres_method,point_adjust=False,composite_best_f1=False,return_auc=True,**thres_args
                )

                thres, pred_labels_fpa, avg_prec, auroc_pa ,test_anom_frac= threshold_and_predict(
                    score, gt, true_events=true_events, logger=None,
                    thres_method=thres_method,point_adjust=True,composite_best_f1=False,return_auc=True,**thres_args
                )

                thres, pred_labels_fc, avg_prec, auroc_c ,test_anom_frac= threshold_and_predict(
                    score, gt, true_events=true_events, logger=None,
                    thres_method=thres_method,point_adjust=False,composite_best_f1=True,return_auc=True,**thres_args
)
                eval_metrics = evaluate_metrics(gt, pred_labels, auroc, pred_labels_fpa, pred_labels_fc, auroc_pa, auroc_c, seq_len=seq_len)

                # Store metrics
                method_metrics[f'{score_type}_metrics_{thres_method}'] = eval_metrics
                
            else:
                # Apply threshold and predict
                thres, pred_labels, avg_prec, auroc ,test_anom_frac= threshold_and_predict(
                    score, gt, true_events=true_events, logger=None,
                    thres_method=thres_method,return_auc=True,**thres_args
                )
                # Evaluate metrics
                eval_metrics = evaluate_metrics(gt, pred_labels, auroc, seq_len=seq_len)
                print(f"thres:{thres}")
                
            # Store metrics
                method_metrics[f'{score_type}_metrics_{thres_method}'] = eval_metrics
            
        metrics[f'metrics_{thres_method}'] = method_metrics

    return metrics, test_anom_frac

def plot_loss(metrics, save_dir, file_name='train_loss_plot.png', mode="Train"):

    # Check if input is a dictionary
    if isinstance(metrics, dict):
        # Extract iterations and loss from the metrics dictionary
        if 'iters' not in metrics or 'loss' not in metrics:
            raise ValueError("Dictionary must contain 'iters' and 'loss' keys.")
        iters = metrics['iters']
        loss = metrics['loss']
    
    # If input is a list, use the list as the y-axis and steps as the x-axis
    elif isinstance(metrics, np.ndarray):
        loss = metrics
        iters = list(range(1, len(loss) + 1)) # X-axis as the number of steps
    
    else:
        raise TypeError("Input must be either a dictionary or a list.")
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(iters, loss, label=f'{mode} Loss', color='blue', linewidth=2)
    
    # Adding labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'{mode} Loss over Iterations')
    plt.legend()
    plt.grid(True)

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the plot to the specified directory
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)

    # Close the plot to free up memory
    plt.close()

    print(f"Plot saved to {save_path}")

def plot_memory(metrics,save_dir, file_name='train_memory_plot.png', mode='Train', avg_reserved_memory=None):
    # Check if input is a dictionary
    if isinstance(metrics, dict):
        # Extract iterations and loss from the metrics dictionary
        if 'iters' not in metrics or 'allocated_memory' not in metrics or 'reserved_memory' not in metrics:
            raise ValueError("Dictionary must contain 'iters' and 'allocated_memory' and 'reserved_memory' keys.")
        iters = metrics['iters']
        allocated_memory = metrics['allocated_memory']
        reserved_memory = metrics['reserved_memory']
        
    elif isinstance(metrics, np.ndarray):
        allocated_memory = metrics
        iters = list(range(1, len(allocated_memory) + 1))
        reserved_memory = avg_reserved_memory

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(iters, allocated_memory, label='Allocated memory', color='orange', linewidth=1)
    if reserved_memory is not None:
        plt.plot(iters, reserved_memory, label='Reserved memory', color='purple', linewidth=2)

    # Adding labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Memory usage in MB')
    plt.title(f'{mode}: Memory usage over Iterations')
    plt.legend()
    plt.grid(True)

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the plot to the specified directory
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)

    # Close the plot to free up memory
    plt.close()

    print(f"Plot saved to {save_path}")