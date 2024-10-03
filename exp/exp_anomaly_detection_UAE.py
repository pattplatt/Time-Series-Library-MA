from data_provider.data_factory import data_provider, get_events
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, write_to_csv, get_dynamic_scores, threshold_and_predict, get_gaussian_kernel_scores, compute_metrics
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
import subprocess

warnings.filterwarnings('ignore')

def fit_one_channel(setting, channel_num, path, args, device):
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')

    model_object = Exp_Anomaly_Detection_UAE(args)
    model = model_object._build_model()
    criterion = model_object._select_criterion()
    model_optim = model_object._select_optimizer(model)

    time_now = time.time()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    model.to(device)
    model.train()
    total_iters = 0
    metrics = {'epoch': [], 'iters': [], 'loss': []}

    # Initialize validation iterator outside the epoch loop
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

        # Validation check (process a single batch per epoch)
        model.eval()
        print(f"Validating channel {channel_num}")

        with torch.no_grad():
            try:
                # Fetch the next validation batch
                batch_x_val, _ = next(vali_iter)
            except StopIteration:
                # If we reach the end of the validation set, reinitialize the iterator
                vali_iter = iter(vali_loader)
                batch_x_val, _ = next(vali_iter)

            batch_x_val = batch_x_val[:, :, channel_num:channel_num+1].float().to(device)
            outputs = model(batch_x_val, None, None, None)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            vali_loss = criterion(outputs, batch_x_val)
            val_loss.append(vali_loss.item())
            outputs = outputs.detach()

        train_loss_avg = np.average(train_loss)
        val_loss_avg = np.average(val_loss)
        model.train()
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, args)

    # Save the model
    model_filename = os.path.join(path, f"model_channel_{channel_num}.pth")
    torch.save(model.state_dict(), model_filename)
    #print(f"Validation loss avg: {val_loss_avg:.7f}")

    del batch_x
    del model
    torch.cuda.empty_cache()

    return train_loss_avg, val_loss_avg, channel_num

@torch.no_grad()
def predict_one_channel(setting, channel_num, saved_array_path, args, device):
    time_now = time.time()
    anomaly_criterion = nn.MSELoss(reduction='none')
    criterion = nn.MSELoss()
    
    train_data, train_loader = data_provider(args,flag='train')
    test_data, test_loader = data_provider(args,flag='test')  
    
    model_object=Exp_Anomaly_Detection_UAE(args)
    model = model_object._build_model()
    # Build and load model
    model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, f'model_channel_{channel_num}.pth'), map_location=device))
    model.to(device)
    model.eval()
    
    # Get data loaders
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
            #print("score after mean", score.shape)

            score = score.view(-1).cpu().numpy()
            #print("score after flatten", score.shape)                
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
    #combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    # Save arrays
    np.save(os.path.join(saved_array_path, f"train_energy_channel_{channel_num}.npy"), train_energy)
    print(f"Saved train_energy{channel_num}.npy")

    np.save(os.path.join(saved_array_path, f"test_energy_channel_{channel_num}.npy"), test_energy)
    print(f"Saved test_energy_channel_{channel_num}.npy")

    # Clean up
    del train_data, train_loader, test_data, test_loader, test_energy, train_energy, model
    torch.cuda.empty_cache()

    return channel_num, test_loss, test_labels

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

    def get_events(self, y_test):
        events = get_events(y_test)
        return events

    def train(self, setting):
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
            args = (
                setting, 
                channel_num, 
                path, 
                self.args, 
                self.device,
            )
            results.append(pool.apply_async(fit_one_channel, args=args))
 
        return_values = [result.get() for result in results]
        pool.close()
        pool.join()

        # Collect and assign models and losses
        train_loss, val_loss= [], []
        
        for return_value in return_values:
            train_l, val_l, channel_num = return_value
            #self.model[channel_num] = model
            train_loss.append(train_l)
            val_loss.append(val_l)
        train_duration = time.time() - now.timestamp()
        return self.model, train_loss, val_loss, train_duration

    def test(self, setting,train_losses, vali_losses, train_duration, test=0):

        self.channels_ = self.args.enc_in
        channel_results = {'channel': []}
        
        saved_array_path = os.path.join(self.args.checkpoints, setting,"test_arrays")
        if not os.path.exists(saved_array_path):
            os.makedirs(saved_array_path)
        
        test_results_path = './test_results/'+ setting
        if not os.path.exists(test_results_path):
            os.makedirs(test_results_path)

        # Number of channels
        num_channels = self.args.enc_in
        test_losses =[]
        memory_stats = []
        
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=30)
        results=[]
        test_start_time = time.time()
        for channel_num in range(self.args.enc_in):
            args = (
                setting, 
                channel_num, 
                saved_array_path,
                self.args,
                self.device
            )
            print("Testing univariate model on channel number %i" % channel_num)
            results.append(pool.apply_async(predict_one_channel, args=args))
            
            # Get memory usage
            mem_info = subprocess.run(["grep", "MemTotal\|MemFree\|MemAvailable", "/proc/meminfo"], capture_output=True, text=True).stdout
            mem_info = {line.split(":")[0]: int(line.split()[1]) for line in mem_info.splitlines()}
            free_memory_kb = mem_info["MemFree"]
            available_memory_kb = mem_info["MemAvailable"]
            # Convert to MB/GB
            available_memory_gb = available_memory_kb / 1024 / 1024
            #print(f"Available Memory: {available_memory_gb:.6f} GB")
            memory_stats.append(available_memory_gb)
            
        return_values = [result.get() for result in results]
        pool.close()
        pool.join()
        
        for return_value in return_values:
            channel_num, test_loss_channel, test_labels = return_value
            channel_results['channel'].append(channel_num)
            test_losses.append(test_loss_channel)
        self.test_labels = test_labels
        
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        total_time = train_duration + test_duration        
        train_energy_arrays = []
        combined_test_energy_arrays = []

        # Load arrays from disk
        for i in range(self.args.enc_in):
            train_energy_array_filename = f"train_energy_channel_{i}.npy"  # Corrected filename formatting            
            train_energy = np.load(os.path.join(saved_array_path, train_energy_array_filename))
            train_energy_arrays.append(train_energy)
            print(f"Loaded {train_energy_array_filename} with shape {train_energy.shape}")
            
            combined_test_array_filename = f"test_energy_channel_{i}.npy"  # Corrected filename formatting
            test_energy = np.load(os.path.join(saved_array_path, combined_test_array_filename))
            combined_test_energy_arrays.append(test_energy)
            print(f"Loaded {combined_test_array_filename} with shape {test_energy.shape}")            
        #Take anomaly scores from each row and combine them with average into single anomaly score
        stacked_train_energy_array = np.stack(train_energy_arrays, axis=0)
        stacked_combined_test_array = np.stack(combined_test_energy_arrays, axis=0)
        print("stacked_train_energy_array:",stacked_train_energy_array.shape)
        print("stacked_combined_test_array:",stacked_combined_test_array.shape)

        avg_train_energy = np.mean(stacked_train_energy_array,0)
        avg_test_energy = np.mean(stacked_combined_test_array,0)
        print("avg_train_energy:",avg_train_energy.shape)
        print("avg_test_energy:",avg_test_energy.shape)
        
        #combined_energy = np.concatenate([avg_train_energy, avg_test_energy], axis=0)

        test_labels = self.test_labels
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        true_events=self.get_events(gt)
        #get dynamic kernel scores
        score_t_test_dyn, _, score_t_train_dyn, _ = get_dynamic_scores(None, None, avg_train_energy,avg_test_energy, long_window=self.args.d_score_long_window, short_window=self.args.d_score_short_window)
        #get dynamic scores
        score_t_test_dyn_gauss_conv, _ = get_gaussian_kernel_scores(avg_test_energy,None,self.args.kernel_sigma)

        score_t_train_dyn_gauss_conv, _ = get_gaussian_kernel_scores(avg_train_energy,None,self.args.kernel_sigma)

        #dyn_scores_combined = np.concatenate([score_t_train_dyn, score_t_test_dyn], axis=0)

        metrics = compute_metrics(avg_test_energy, gt, true_events,score_t_test_dyn, score_t_test_dyn_gauss_conv, self.args.seq_len , avg_train_energy, score_t_train_dyn, score_t_train_dyn_gauss_conv)
        #metrics = compute_metrics(test_scores, gt, true_events,score_t_test_dyn, score_t_dyn_gauss_conv, self.args.seq_len , train_scores, score_t_train_dyn, train_score_t_dyn_gauss_conv)
        
        avg_test_loss = np.mean(test_losses)
        avg_train_loss = np.mean(train_losses)
        avg_vali_loss = np.mean(vali_losses)

        test_results_path = os.path.join('./test_results/', setting)
        if not os.path.exists(test_results_path):
            os.makedirs(test_results_path)
        export_memory_usage = False
        write_to_csv(
            model_name=self.args.model,
            model_id=self.args.model_id,
            avg_train_loss=avg_train_loss,
            avg_vali_loss=avg_vali_loss,
            avg_test_loss=avg_test_loss,
            seq_len=self.args.seq_len,
            d_model=self.args.d_model,
            enc_in=self.args.enc_in,
            e_layers=self.args.e_layers,
            dec_in=self.args.dec_in,
            d_layers=self.args.d_layers,
            c_out=self.args.c_out,
            d_ff=self.args.d_ff,
            n_heads=self.args.n_heads,
            long_window=self.args.d_score_long_window, 
            short_window=self.args.d_score_short_window,
            kernel_sigma=self.args.kernel_sigma,
            train_epochs=self.args.train_epochs,
            learning_rate=self.args.learning_rate,
            anomaly_ratio=self.args.anomaly_ratio,
            embed=self.args.embed,
            total_time=total_time,
            train_duration=train_duration,
            test_duration=test_duration,
            metrics=metrics,
            test_results_path=test_results_path,
            setting=setting,
            benchmark_id = self.args.benchmark_id,
            export_memory_usage=export_memory_usage
        )

        return