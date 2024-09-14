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
import subprocess

warnings.filterwarnings('ignore')

class Exp_Anomaly_Detection_UAE(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_UAE, self).__init__(args)

    def _build_model(self):
        args_model = copy.deepcopy(self.args)
        args_model.enc_in = 1
        args_model.c_out = 1
        
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
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=30)
        criterion = self._select_criterion()

        results = []        
        for channel_num in range(self.args.enc_in):
            self.model = self._build_model().cpu()
            model_optim = self._select_optimizer(self.model)
            
            args = (
                train_loader, 
                setting, 
                channel_num, 
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
    def fit_one_channel(train_loader, setting, channel_num, path, args, model_optim, criterion, model, device,vali_loader):
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
    @staticmethod
    def predict_one_channel(train_loader, test_loader, setting, channel_num, saved_array_path, args, anomaly_criterion, model, device):
        time_now = time.time()
        model.to(device)
        model.eval()
        test_loss = []
        
        # Get number of samples in train_data and test_data
        n_train_samples = len(train_data)
        n_test_samples = len(test_data)

        # Preallocate arrays
        train_energy = np.zeros(n_train_samples*args.seq_len)
        test_energy = np.zeros(n_test_samples*args.seq_len)
        test_labels = np.zeros(n_test_samples*args.seq_len)

        # Process train set
        idx = 0
        with torch.no_grad():
            for batch_x, _ in train_loader:
                batch_size = batch_x.size(0)
                batch_x = batch_x[:, :, channel_num:channel_num+1].float().to(device)
                outputs = model(batch_x, None, None, None)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                score = anomaly_criterion(batch_x, outputs)
                score = torch.mean(score, dim=2)    
                print("score after mean", score.shape)
                
                score = score.view(-1).cpu().numpy()
                print("score after flatten", score.shape)                
                train_energy[idx:idx + batch_size*args.seq_len] = score
                idx += batch_size
                del batch_x, outputs, score  # Free up memory
                torch.cuda.empty_cache()

        # Process test set
        idx = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_size = batch_x.size(0)
                batch_x = batch_x[:, :, channel_num:channel_num+1].float().to(device)
                outputs = model(batch_x, None, None, None)
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                
                score = anomaly_criterion(batch_x, outputs)
                #print("score 1", score.shape)
                score = torch.mean(score, dim=2)    
                #print("score after mean", score.shape)
                score = score.view(-1).cpu().numpy()
                #print("score after flatten", score.shape)
                
                test_loss.append(criterion(outputs.cpu(), batch_x.cpu()).item())                
                test_energy[idx:idx + batch_size*args.seq_len] = score
                #print("batch_y:",batch_y.shape)
                test_labels[idx:idx + batch_size*args.seq_len] = batch_y.view(-1).cpu().numpy()
                idx += batch_size
                del batch_x, batch_y, outputs, score  # Free up memory
                torch.cuda.empty_cache()

        # Combine train and test energies
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        # Save arrays
        np.save(os.path.join(array_path, f"combined_energy_channel_{channel_num}.npy"), combined_energy)
        print(f"Saved combined_energy_channel_{channel_num}.npy")

        np.save(os.path.join(array_path, f"test_energy_channel_{channel_num}.npy"), test_energy)
        print(f"Saved test_energy_channel_{channel_num}.npy")

        # Clean up
        del train_data, train_loader, test_data, test_loader, train_energy, test_energy, combined_energy, model
        torch.cuda.empty_cache()

        return channel_num, test_loss, test_labels

    def test(self, setting,train_loss, vali_loss, test_loss,model,seq_len ,d_model,e_layers,d_ff,n_heads,train_epochs,loss,learning_rate,anomaly_ratio,embed,train_duration, test=0):
        
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')        
        self.channels_ = self.args.enc_in
        channel_results = {'channel': []}
        
        saved_array_path = os.path.join(self.args.checkpoints, setting,"test_arrays")
        if not os.path.exists(saved_array_path):
            os.makedirs(saved_array_path)
        
        test_results_path = './test_results/'+ setting
        if not os.path.exists(test_results_path):
            os.makedirs(test_results_path)

        device = self.device
        anomaly_criterion = nn.MSELoss(reduce=False)

        # Number of channels
        num_channels = self.args.enc_in
        test_losses =[]
        memory_stats = []
        
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=30)
        results=[]
        test_start_time = time.time()
        for channel_num in range(self.args.enc_in):
            model = self._build_model().cpu()
            model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'model_channel_'+ str(channel_num) +'.pth')))
            #def predict_one_channel(train_loader, test_loader, setting, channel_num, saved_array_path,args, model_optim, criterion, model, device):
            args = (
                train_loader,
                test_loader,
                setting, 
                channel_num, 
                saved_array_path, 
                self.args,
                anomaly_criterion, 
                model, 
                self.device,
            )
            print("Testing univariate model on channel number %i" % channel_num)
            results.append(pool.apply_async(self.predict_one_channel, args=args))
            
            return_values = [result.get() for result in results]

            pool.close()
            pool.terminate()
            pool.join()

            #print("Memory allocated before: ", torch.cuda.memory_allocated())
            #channel_num, test_loss_channel,test_labels = self.predict_one_channel(setting, channel_num, self.args, self.device, saved_array_path)
            channel_results['channel'].append(channel_num)
            test_losses.append(test_loss_channel)

            self.model.cpu()
            del self.model, channel_num, test_loss_channel
            torch.cuda.empty_cache()
            
            #print("Memory allocated after: ", torch.cuda.memory_allocated())
            self.test_labels = test_labels

            # Get memory usage
            mem_info = subprocess.run(["grep", "MemTotal\|MemFree\|MemAvailable", "/proc/meminfo"], capture_output=True, text=True).stdout
            mem_info = {line.split(":")[0]: int(line.split()[1]) for line in mem_info.splitlines()}

            free_memory_kb = mem_info["MemFree"]
            available_memory_kb = mem_info["MemAvailable"]

            # Convert to MB/GB
            available_memory_gb = available_memory_kb / 1024 / 1024

            # Print the results
            print(f"Available Memory: {available_memory_gb:.6f} GB")
            memory_stats.append(available_memory_gb)
            
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        total_time = train_duration + test_duration        
        combined_energy_arrays = []
        combined_test_energy_arrays = []

        # Load arrays from disk
        for i in range(self.args.enc_in):
            combined_energy_array_filename = f"combined_energy_channel_{i}.npy"  # Corrected filename formatting            
            combined_energy = np.load(os.path.join(saved_array_path, combined_energy_array_filename))
            combined_energy_arrays.append(combined_energy)
            print(f"Loaded {combined_energy_array_filename} with shape {combined_energy.shape}")
            
            combined_test_array_filename = f"test_energy_channel_{i}.npy"  # Corrected filename formatting
            test_energy = np.load(os.path.join(saved_array_path, combined_test_array_filename))
            combined_test_energy_arrays.append(test_energy)
            print(f"Loaded {combined_test_array_filename} with shape {test_energy.shape}")            
            
        #Take anomaly scores from each row and combine them with average into single anomaly score
        stacked_combined_energy_array = np.stack(combined_energy_arrays, axis=0)
        stacked_combined_test_array = np.stack(combined_test_energy_arrays, axis=0)
        print("stacked_combined_energy_array:",stacked_combined_energy_array.shape)
        print("stacked_combined_test_array:",stacked_combined_test_array.shape)
        
        avg_combined_energy = np.mean(stacked_combined_energy_array,0)
        avg_test_energy = np.mean(stacked_combined_test_array,0)
        print("avg_combined_energy:",avg_combined_energy.shape)
        print("avg_test_energy:",avg_test_energy.shape)

        threshold = np.percentile(avg_combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (avg_test_energy > threshold).astype(int)

        test_labels = self.test_labels
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

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

        avg_test_loss= np.mean(torch.tensor(test_losses).cpu().numpy())

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

        file_path = os.path.join(test_results_path, "results_")
        file_exists = os.path.isfile(file_path)
             
        with open("Memory"+ ".csv", 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(memory_stats)        
            
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