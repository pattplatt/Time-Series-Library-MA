from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, get_events, get_composite_fscore_raw
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
import copy
import torch.multiprocessing as mp
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv

warnings.filterwarnings('ignore')

class Exp_Anomaly_Detection_UAE(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_UAE, self).__init__(args)

    def _build_model(self):
        args_model = copy.deepcopy(self.args)
        args_model.enc_in = 1
        
        self.model = self.model_dict[self.args.model].Model(args_model).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return self.model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        #test_data, test_loader = self._get_data(flag='test')
        
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M")
        
        folder_path = './test_results/HTTP_benchmarkv3/'
        os.makedirs(folder_path, exist_ok=True)
            
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
            
        ctx = mp.get_context('spawn')

        pool = ctx.Pool(processes=10)

        results = []
        
        for channel_num in range(self.args.enc_in):
            self.model = self._build_model().cpu()
            model_optim = self._select_optimizer(self.model)
            criterion = self._select_criterion()
            
            args = (
                train_loader, 
                setting, 
                channel_num, 
                folder_path, 
                path, 
                self.args, 
                model_optim, 
                criterion, 
                self.model, 
                self.device,
                vali_loader
            )
            results.append(pool.apply_async(self.fit_one_channel, args=args))
        return_values = [result.get() for result in results]
        pool.close()
        pool.terminate()
        pool.join()
        
        # Collect and assign models and losses
        #self.model = [None] * self.args.enc_in
        train_loss, val_loss, test_loss = [], [], []
        
        for return_value in return_values:
            train_l, val_l, test_l, channel_num = return_value
            #self.model[channel_num] = model
            train_loss.append(train_l)
            val_loss.append(val_l)
            test_loss.append(test_l)
        train_duration = time.time() - now.timestamp()
        return self.model, train_loss, val_loss, test_loss, train_duration

    @staticmethod
    def fit_one_channel(train_loader, setting, channel_num, folder_path, path, args, model_optim, criterion, model, device,vali_loader):
        time_now = time.time()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        model.to(device)
        model.train()
        total_iters = 0
        metrics = {'epoch': [], 'iters': [], 'loss': []}
        vali_iter = iter(vali_loader)
        print("Training univariate model on channel number %i" % channel_num)
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            val_loss = []

            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x[:, :, channel_num:channel_num+1].float().to(device)

                model_optim.zero_grad()
                outputs = model(batch_x, None, None, None)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"iters: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    metrics['epoch'].append(epoch + 1)
                    metrics['iters'].append(total_iters)
                    metrics['loss'].append(loss.item())

                iter_count += 1
                total_iters += 1

            # Validation check (no need to loop over entire validation set)
            model.eval()

            #try:
                # Fetch the next validation batch
                #batch_x_val, _ = next(vali_iter)
            #except StopIteration:
                # If we reach the end of the validation set, reinitialize the iterator
                #vali_iter = iter(vali_loader)
                #batch_x_val, _ = next(vali_iter)
            print(f"Validating channel {channel_num}")
            # Process validation batch
            
            with torch.no_grad():
                for i, (batch_x_val, _) in enumerate(vali_loader):
                    batch_x_val = batch_x_val[:, :, channel_num:channel_num+1].float().to(device)
                    outputs = model(batch_x_val, None, None, None)
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, :, f_dim:]
                    vali_loss = criterion(outputs, batch_x_val)
                    val_loss.append(vali_loss.item())
                    outputs = outputs.detach() 

            train_loss_avg = np.average(train_loss)
            val_loss_avg = np.average(val_loss)
            test_loss = 0  # Placeholder for test logic
            model.train()
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, args)

        # Save the model
        model_filename = os.path.join(path, f"model_channel_{channel_num}.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"Validation loss avg: {val_loss_avg:.7f}")

        del batch_x, _
        del model
        torch.cuda.empty_cache()
        
        return train_loss_avg, val_loss_avg, test_loss, channel_num

    @torch.no_grad()
    def predict_one_channel(self,setting, channel_num, args, device):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        self.model = self._build_model()
        time_now = time.time()

        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'model_channel_'+ str(channel_num) +'.pth')))
        self.model.to(device)
        criterion = self._select_criterion()
        anomaly_criterion = nn.MSELoss(reduce=False)
        self.model.eval()
        attens_energy = []
        test_loss = []

        # (1) stastic on the train set
        with torch.no_grad():  
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x[:, :, channel_num:channel_num+1].float().to(device)
                outputs = self.model(batch_x, None, None, None)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                score =anomaly_criterion(batch_x, outputs)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        shapes = [np.array(e).shape for e in attens_energy]
        #print("Shapes of attens_energy elements:", shapes)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x[:, :, channel_num:channel_num+1]
                batch_x = batch_x.float().to(self.device)

                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
                test_loss.append(criterion(outputs,batch_x))
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
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
        del batch_x, batch_y
        del self.model
        torch.cuda.empty_cache()
        
        return channel_num, pred, gt, test_loss

    def test(self, setting,train_loss, vali_loss, test_loss,model,seq_len ,d_model,e_layers,d_ff,n_heads,train_epochs,loss,learning_rate,anomaly_ratio,embed,train_duration, test=0):
        self.channels_ = self.args.enc_in
        test_start_time = time.time()
        channel_results = {'channel': [], 'gt': [], 'pred': []}
        folder_path = './test_results/HTTP_benchmarkv3/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # Number of channels
        num_channels = self.args.enc_in
        test_losses =[]
        # Arrays to store metrics
        accuracies = np.zeros(num_channels)
        precisions = np.zeros(num_channels)
        recalls = np.zeros(num_channels)
        f1s = np.zeros(num_channels)

        for channel_num in range(self.args.enc_in):
            print("Testing univariate model on channel number %i" % channel_num)
            channel_num, pred, gt,test_loss_channel = self.predict_one_channel(setting, channel_num, self.args, self.device)
            channel_results['channel'].append(channel_num)
            channel_results['gt'].append(pred)
            channel_results['pred'].append(gt)
            test_losses.append(test_loss_channel)
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        total_time = train_duration + test_duration

        for i, channel in enumerate(channel_results['channel']):
            gt = channel_results['gt'][i]
            pred = channel_results['pred'][i]

            # General Accuracy and Confusion Matrix
            accuracy = accuracy_score(gt, pred)
            cm = confusion_matrix(gt, pred)

            # Normalize the confusion matrix by sequence length
            cm_normalized = np.floor(cm / seq_len)

            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

            # Store results in arrays
            accuracies[i] = accuracy
            precisions[i] = precision
            recalls[i] = recall
            f1s[i] = f_score

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

        # Compute average metrics
        average_accuracy = np.mean(accuracies)
        average_precision = np.mean(precisions)
        average_recall = np.mean(recalls) 
        average_f1 = np.mean(f1s)
        avg_test_loss= np.mean(torch.tensor(test_losses).cpu().numpy())

        print("\nAverage Metrics across all channels:")
        print(f"  Average Accuracy: {average_accuracy:.2f}")
        print(f"  Average Precision: {average_precision:.2f}")
        print(f"  Average Recall: {average_recall:.2f}")
        print(f"  Average F1: {average_f1:.2f}")

        # Get the current date and time
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M")

        avg_train_loss = np.mean(train_loss)
        avg_vali_loss = np.mean(vali_loss)

        parameters_header = ["Model", "avg_train_loss", "avg_vali_loss", "avg_test_loss", "Seq len", "Dim Model", "Enc-layers", "Dim-ff", "n_heads", "Train epochs", "Learning rate", "Anomaly ratio", "embed", "Total Duration (s)", "Train Duration (s)", "Test Duration (s)","Test Duration (min)"]
        parameters = [self.args.model, f"{avg_train_loss:.6f}", f"{avg_vali_loss:.6f}", f"{avg_test_loss:.6f}", seq_len, d_model, e_layers, d_ff, n_heads, train_epochs, learning_rate, anomaly_ratio, embed,f"{total_time:.2f}", f"{train_duration:.2f}", f"{test_duration:.2f}",f"{(total_time/60):.2f}"]

        metrics_header = ["Model","F-Metric","Accuracy", "Precison","Recall","F-1_x", "CM"]
        metrics_f1 = [self.args.model,"Default F1", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f_score:.4f}", f"{cm}"]
        metrics_fpa = [self.args.model, "Fpa", f"{accuracy_point_adjusted:.4f}", f"{precision_point_adjusted:.4f}", f"{recall_point_adjusted:.4f}", f"{f_score_point_adjusted:.4f}", f"{cm_point_adjusted}"]
        metrics_fc = [self.args.model, "Fc", f"{accuracy_point_adjusted:.4f}", f"{prec_t:.4f}", f"{rec_e:.4f}", f"{fscore_c:.4f}", f"-"]

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