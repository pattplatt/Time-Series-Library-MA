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

def get_composite_fscore_raw(pred_labels, true_events, y_test, return_prec_rec=False):
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    total_length = len(y_test)
    # Get Negative Intervals
    negative_intervals = get_negative_intervals(true_events, total_length)
    tn = np.sum([not pred_labels[start:end + 1].any() for start, end in negative_intervals])
    fp = np.sum([pred_labels[start:end + 1].any() for start, end in negative_intervals])
    fn = len(true_events) - tp
    acc_c = (tp+tn)/(tp+tn+fn+fp)
    rec_e = tp/(tp + fn)
    prec_t = precision_score(y_test, pred_labels)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c, acc_c
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
        if score_t_test_and_train is not None:
            print("using score_t_test_and_train")
            opt_thres = np.nanpercentile(
                score_t_test_and_train, 100 * (1 - test_anom_frac), interpolation="higher"
            )
        else:
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

def evaluate_metrics(gt, pred, auroc,seq_len=1,export_memory_usage=False):
    """
    Evaluate various metrics given ground truth labels and predictions.

    Parameters:
    - gt: array-like of shape (n_samples,), Ground truth labels.
    - pred: array-like of shape (n_samples,), Predicted labels.
    - export_memory_usage: bool, default False. If True, exports memory usage statistics to 'Memory.csv'.
    - seq_len: int, default 1. Sequence length for normalizing confusion matrix.

    Returns:
    - metrics: dict, containing computed metrics.
    """
    # General Accuracy and Confusion Matrix
    accuracy = accuracy_score(gt, pred)
    cm = confusion_matrix(gt, pred)

    # Avoid division by zero in case seq_len is zero
    if seq_len == 0:
        print("Warning: seq_len is zero. Normalized confusion matrix cannot be computed.")
        cm_normalized = cm
    else:
        cm_normalized = np.floor(cm / seq_len)

    precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')

    #print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
    #print("Confusion Matrix (Normalized):")
    #print(cm_normalized)

    # Point-adjusted F1 Score
    try:
        labels_point_adjusted, anomaly_preds_point_adjusted = adjustment(gt, pred)
        accuracy_point_adjusted = accuracy_score(labels_point_adjusted, anomaly_preds_point_adjusted)
        precision_point_adjusted, recall_point_adjusted, f_score_point_adjusted, _ = precision_recall_fscore_support(
            labels_point_adjusted, anomaly_preds_point_adjusted, average='binary')
        cm_point_adjusted = confusion_matrix(labels_point_adjusted, anomaly_preds_point_adjusted)

        #print("Adjusted Metrics:")
        #print("Accuracy: {:0.4f}, Precision: {:0.4f}, Recall: {:0.4f}, F-score: {:0.4f}".format(accuracy_point_adjusted, precision_point_adjusted, recall_point_adjusted, f_score_point_adjusted))
        #print("Confusion Matrix (Point-adjusted):")
        #print(cm_point_adjusted)
    except NameError:
        print("Error: 'adjustment' function is not defined. Skipping point-adjusted metrics.")
        accuracy_point_adjusted = precision_point_adjusted = recall_point_adjusted = f_score_point_adjusted = None
        cm_point_adjusted = None

    # Composite F1 Score
    try:
        true_events = get_events(gt)
        prec_t, rec_e, fscore_c, acc_c= get_composite_fscore_raw(pred, true_events, gt, return_prec_rec=True)
        #print("Composite F1 Score:")
        #print("Acc_c: {:0.4f},Prec_t: {:0.4f}, Rec_e: {:0.4f}, Fscore_c: {:0.4f}".format(acc_c, prec_t, rec_e, fscore_c))
    except NameError:
        print("Error: 'get_events' or 'get_composite_fscore_raw' function is not defined. Skipping composite F1 score.")
        prec_t = rec_e = fscore_c = None

    # Optionally, export memory usage
    if export_memory_usage:
        try:
            # Collect memory stats here
            memory_stats = get_memory_stats()  # You need to define get_memory_stats()
            # Write to file
            with open("Memory.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(memory_stats)
        except NameError:
            print("Error: 'get_memory_stats' function is not defined. Cannot export memory usage.")

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
        'acc_c': acc_c,
        'prec_t': prec_t,
        'rec_e': rec_e,
        'fscore_c': fscore_c,
        'auroc':auroc,
    }

    return metrics

def write_to_csv(
    model_name, model_id,avg_train_loss, avg_vali_loss, avg_test_loss, seq_len, d_model,enc_in, e_layers,dec_in,d_layers,c_out,
    d_ff, n_heads, long_window, short_window, kernel_sigma, train_epochs, learning_rate, anomaly_ratio, embed, total_time, train_duration,
    test_duration, metrics, test_results_path, setting, export_memory_usage=False, memory_stats=None
):
    """
    Writes parameters and metrics to CSV files.

    Parameters:
    - model_name (str): Name of the model.
    - avg_train_loss (float): Average training loss.
    - avg_vali_loss (float): Average validation loss.
    - avg_test_loss (float): Average test loss.
    - seq_len (int): Sequence length.
    - d_model (int): Model dimension.
    - e_layers (int): Number of encoder layers.
    - d_ff (int): Dimension of feedforward network.
    - n_heads (int): Number of attention heads.
    - train_epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate.
    - anomaly_ratio (float): Anomaly ratio.
    - embed (str): Embedding type.
    - total_time (float): Total duration in seconds.
    - train_duration (float): Training duration in seconds.
    - test_duration (float): Testing duration in seconds.
    - metrics (dict or list): Dictionary or list containing evaluation metrics.
    - test_results_path (str): Path to save test results.
    - setting (str): Current setting or configuration name.
    - export_memory_usage (bool, optional): If True, writes memory stats to 'Memory.csv'. Defaults to False.
    - memory_stats (list, optional): Memory statistics to write. Required if export_memory_usage is True.

    Returns:
    - None
    """
    
    # Parameters header and data
    parameters_header = [
        "Model", "avg_train_loss", "avg_vali_loss", "avg_test_loss", "seq_len", "d_model", "enc_in","e_layers", "dec_in", "d_layers", "c_out","d_ff","n_heads","long_window","short_window","kernel_sigma","train_epochs","learning_rate","anomaly_ratio", "embed", "Total Duration (min)", "Train Duration (min)", "Test Duration (min)"]
    parameters = [
        model_name, f"{avg_train_loss:.6f}", f"{avg_vali_loss:.6f}", f"{avg_test_loss:.6f}", seq_len, d_model, enc_in,e_layers, dec_in,d_layers,c_out,d_ff,n_heads, long_window, short_window, kernel_sigma, train_epochs, learning_rate, anomaly_ratio, embed, f"{(total_time / 60):.2f}",f"{(train_duration/60):.2f}", f"{(test_duration/60):.2f}"]

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
                f"{metrics_dict['f_score_point_adjusted']:.4f}",f"{metrics_dict.get('auroc', 'N/A'):.4f}", f"{metrics_dict['cm_point_adjusted']}"
            ]
        else:
            metrics_fpa = [model_name, "Fpa", "N/A", "N/A", "N/A", "N/A", "N/A"]
        writer.writerow(metrics_fpa)

        # Composite F1 metrics
        if metrics_dict.get('fscore_c') is not None:
            metrics_fc = [
                model_name, "Fc", f"{metrics_dict['acc_c']:.4f}", f"{metrics_dict['prec_t']:.4f}", f"{metrics_dict['rec_e']:.4f}", f"{metrics_dict['fscore_c']:.4f}",f"{metrics_dict.get('auroc', 'N/A'):.4f}", "-"
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
    
    write_f_score_metrics(model_name, model_id, metrics, test_results_path)
    # Optionally, write memory stats
    if export_memory_usage and memory_stats is not None:
        with open("Memory.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(memory_stats)
            
def write_f_score_metrics(model_name, model_id, metrics, test_results_path):
    """
    Writes specified metrics to a transposed CSV file, with metrics as rows and models as columns.
    Computes mean scores per threshold method and overall means.

    Parameters:
    - model_name (str): Name of the model.
    - model_id (str): Identifier for the model.
    - metrics (dict): Dictionary containing evaluation metrics.
    - test_results_path (str): Path to save test results.

    Returns:
    - None
    """
    # Get the parent directory of test_results_path
    parent_dir = os.path.dirname(test_results_path)

    # Construct the path for aggregate_metrics.csv in the parent directory
    metrics_csv_file = os.path.join(parent_dir, "aggregate_metrics.csv")
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

def compute_metrics(test_energy, gt, true_events, score_t_test_dyn, score_t_test_dyn_gauss_conv, seq_len,train_energy=None, score_t_train_dyn=None,score_t_train_dyn_gauss_conv=None):
    """
    Compute metrics using different thresholding methods and scoring functions.

    Args:
        test_energy (np.array): The default anomaly scores for the test data.
        gt (np.array): Ground truth labels for the test data.
        true_events (dict): Dictionary of true anomaly events.
        score_t_test_dyn (np.array, optional): Dynamic scores for the test data.
        score_t_test_dyn_gauss_conv (np.array, optional): Dynamic Gaussian convolved scores for the test data.
        seq_len (int): Sequence length used in the model.
        train_energy (np.array, optional):  Default anomaly scores from training data.
        score_t_train_dyn (np.array, optional): Dynamic scores for the training data.
        score_t_train_dyn_gauss_conv (np.array, optional): Dynamic Gaussian convolved scores from training data.

    Returns:
        dict: A dictionary containing computed metrics.
    """
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
            if thres_method == 'top_k_time' or (thres_method == 'tail_prob' and score_type == 'default'):
                thres_args['score_t_test_and_train'] = combined_scores[score_type]
            # If combined_scores[score_type] is None, threshold_and_predict should handle it

            # Apply threshold and predict
            thres, pred_labels, avg_prec, auroc = threshold_and_predict(
                score, gt, true_events=true_events, logger=None,
                thres_method=thres_method, return_auc=True, **thres_args
            )

            # Evaluate metrics
            eval_metrics = evaluate_metrics(gt, pred_labels, auroc, seq_len=seq_len)

            # Store metrics
            method_metrics[f'{score_type}_metrics_{thres_method}'] = eval_metrics

        metrics[f'metrics_{thres_method}'] = method_metrics

    return metrics

