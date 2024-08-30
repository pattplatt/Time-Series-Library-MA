from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, get_events, get_composite_fscore_raw
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from datetime import datetime
import csv

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M")
    
        folder_path = './test_results/HTTP_benchmarkv3/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        metrics = {'epoch': [],'iters': [], 'loss': []}

        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        training_start_time=time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        total_iters=0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                total_iters +=1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    metrics['epoch'].append(epoch + 1)
                    metrics['iters'].append(total_iters)
                    metrics['loss'].append(loss.item())

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        train_end_time = time.time()
        train_duration = train_end_time - training_start_time

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        torch.save(metrics, folder_path+"metrics"+"_"+date_time_str+".pth")

        return self.model, train_loss, vali_loss, test_loss, train_duration
    
    def test(self, setting,train_loss, vali_loss, test_loss,model,seq_len ,d_model,e_layers,d_ff,n_heads,train_epochs,loss,learning_rate,anomaly_ratio,embed,train_duration, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        
        test_start_time = time.time()
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/HTTP_benchmarkv3/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                #print(f"outputs.shape:{outputs.shape}")
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        print("train_energy:",len(train_energy))
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        total_time = train_duration + test_duration
        
        pred = np.array(pred)
        gt = np.array(gt)

        # Ensure pred and gt have the same shape
        if pred.shape != gt.shape:
            raise ValueError("Shape mismatch: pred and gt must have the same shape.")

        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        # General Accuracy and Confusion Matrix
        accuracy = accuracy_score(gt, pred)
        cm = confusion_matrix(gt, pred)

        # Normalize the confusion matrix by sequence length
        cm_normalized = np.floor(cm / seq_len)

        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))

        # Compute the confusion matrix
        print("Confusion Matrix (Normalized):")
        print(cm_normalized)
        
        # Point-adjusted F-1
        labels_point_adjusted, anomaly_preds_point_adjusted = adjustment(gt, pred)
        accuracy_point_adjusted = accuracy_score(labels_point_adjusted, anomaly_preds_point_adjusted)
        precision_point_adjusted, recall_point_adjusted, f_score_point_adjusted, support_point_adjusted = precision_recall_fscore_support(labels_point_adjusted, anomaly_preds_point_adjusted, average='binary')
        cm_point_adjusted = confusion_matrix(labels_point_adjusted, anomaly_preds_point_adjusted)

        print("Accuracy_point_adjusted : {:0.4f}, Precision_point_adjusted : {:0.4f}, Recall_point_adjusted : {:0.4f}, F-score_point_adjusted : {:0.4f}".format(accuracy_point_adjusted, precision_point_adjusted, recall_point_adjusted, f_score_point_adjusted))
        print("Confusion Matrix (Point-adjusted):")
        print(cm_point_adjusted)

        # Composite F1
        true_events = get_events(gt)
        prec_t, rec_e, fscore_c = get_composite_fscore_raw(pred, true_events, gt, return_prec_rec=True)
        print("Prec_t: {:0.4f}, Rec_e: {:0.4f}, Fscore_c: {:0.4f}".format(prec_t, rec_e, fscore_c))
        
        # Get the current date and time
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M")
        
        parameters_header = ["Model", "Train loss", "Vali loss", "Test loss", "Seq len", "Dim Model", "Enc-layers", "Dim-ff", "n_heads", "Train epochs", "Learning rate", "Anomaly ratio", "embed", "Total Duration (s)", "Train Duration (s)", "Test Duration (s)","Test Duration (min)"]
        parameters = [model, f"{train_loss:.6f}", f"{vali_loss:.6f}", f"{test_loss:.6f}", seq_len, d_model, e_layers, d_ff, n_heads, train_epochs, learning_rate, anomaly_ratio, embed,f"{total_time:.2f}", f"{train_duration:.2f}", f"{test_duration:.2f}",f"{(total_time/60):.2f}"]

        metrics_header = ["Model","F-Metric","Accuracy", "Precison","Recall","F-1_x", "CM"]
        metrics_f1 = [model,"Default F1", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f_score:.4f}", f"{cm}"]
        metrics_fpa = [model, "Fpa", f"{accuracy_point_adjusted:.4f}", f"{precision_point_adjusted:.4f}", f"{recall_point_adjusted:.4f}", f"{f_score_point_adjusted:.4f}", f"{cm_point_adjusted}"]
        metrics_fc = [model, "Fc", f"{accuracy_point_adjusted:.4f}", f"{prec_t:.4f}", f"{rec_e:.4f}", f"{fscore_c:.4f}", f"-"]

        file_path = os.path.join(folder_path, "results_")
        file_exists = os.path.isfile(file_path)
                      
        with open(file_path + "parameters"+ ".csv", 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(parameters_header)
            writer.writerow(parameters)
            
        with open(file_path + "metrics" + ".csv", 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(metrics_header)
            writer.writerow(metrics_f1)
            writer.writerow(metrics_fpa)
            writer.writerow(metrics_fc)
            
