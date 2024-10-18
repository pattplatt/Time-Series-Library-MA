from data_provider.data_factory import data_provider, get_events
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, write_to_csv, get_dynamic_scores, threshold_and_predict, get_gaussian_kernel_scores, compute_metrics, plot_loss, plot_memory, fit_distributions, get_scores_channelwise
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

    def get_events(self, y_test):
        events = get_events(y_test)
        return events

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
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d_%H-%M")
            
        metrics = {'epoch': [],'iters': [], 'loss': [], 'allocated_memory': [], 'reserved_memory': []}

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
        vali_loss_array = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                torch.cuda.reset_peak_memory_stats()
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
                
                allocated_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)  # In MB
                reserved_memory = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)  # In MB
                
                metrics['allocated_memory'].append(allocated_memory)
                metrics['reserved_memory'].append(reserved_memory)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            vali_loss_array.append(vali_loss)
            test_loss = self.vali(test_data, test_loader, criterion)            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, np.mean(metrics['loss']),np.mean(vali_loss), np.mean(test_loss)))
            early_stopping(np.mean(vali_loss), self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        plot_loss(metrics, os.path.join('./test_results/', setting), 'training_loss_plot.png','Train')
        plot_loss(np.array(vali_loss_array).flatten(), os.path.join('./test_results/', setting), 'vali_loss_plot.png','Validation')
        plot_memory(metrics, os.path.join('./test_results/', setting), 'training_memory_plot.png','Train')

        train_end_time = time.time()
        train_duration = train_end_time - training_start_time
        avg_train_loss = np.mean(metrics['loss'])
        avg_vali_loss = np.mean(vali_loss)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model, avg_train_loss, avg_vali_loss, train_duration, metrics['allocated_memory'], metrics['reserved_memory']
    
    def test(self, setting, avg_train_loss, avg_vali_loss, train_duration, avg_allocated_memory_train, avg_reserved_memory_train, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        test_metrics = {'iters': [], 'loss': [], 'allocated_memory': [], 'reserved_memory': []}

        test_results_path = os.path.join('./test_results/', setting)

        test_start_time = time.time()
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        train_score = []
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        criterion = self._select_criterion()

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                print("outputs:",outputs.shape)
                score = self.anomaly_criterion(batch_x, outputs)
                print("score:",score.shape)
                score = score.detach().cpu().numpy()
                train_score.append(score)

        train_score = np.concatenate(train_score, axis=0)
        #attens_energy=np.array(attens_energy)
        print(f"train_score:{train_score.shape}")
        
        # Get the shape dynamically
        shape = train_score.shape  # This will return a tuple (77, 128, 30, 123)
        # Use variables to represent the dimensions
        dim1, dim2, dim3 = shape
        # Reshape the array: combine the first three dimensions and keep the last dimension
        train_scores_tc = train_score.reshape(dim1 * dim2, dim3)

        #train_energy = np.array(attens_energy)
        print(f"train_scores_tc:{train_scores_tc.shape}")

        # (2) find the threshold
        test_score = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            torch.cuda.reset_peak_memory_stats()
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            score = self.anomaly_criterion(batch_x, outputs)
            score = score.detach().cpu().numpy()
            test_score.append(score)
            test_labels.append(batch_y)
            
            allocated_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)  # In MB
            reserved_memory = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)  # In MB
            
            test_metrics['iters'].append(i)
            test_metrics['loss'].append(criterion(batch_x, outputs).item())
            test_metrics['allocated_memory'].append(allocated_memory)
            test_metrics['reserved_memory'].append(reserved_memory)

        plot_loss(test_metrics, os.path.join('./test_results/', setting), 'test_loss_plot.png','Test')
        plot_memory(test_metrics, os.path.join('./test_results/', setting), 'test_memory_plot.png', 'Test')
        
        test_score = np.concatenate(test_score, axis=0)

        # Get the shape dynamically
        shape = test_score.shape  # This will return a tuple (77, 128, 30, 123)
        # Use variables to represent the dimensions
        dim1, dim2, dim3 = shape
        # Reshape the array: combine the first three dimensions and keep the last dimension
        test_scores_tc = test_score.reshape(dim1 * dim2, dim3)
        
        print(f"test_scores_tc:{test_scores_tc.shape}")
                
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        total_time = train_duration + test_duration

        gt = np.array(gt)
        #get true events from dataset
        true_events=self.get_events(gt)
        
        distributions_dic = fit_distributions(
            distr_par_file="distr_params.pkl",
            distr_names=["univar_gaussian"],
            predictions_dic={
                "train_raw_scores": train_scores_tc
            }
        )
        
        # Compute anomaly scores
        train_ano_scores, val_ano_scores, test_ano_scores, train_prob_scores, val_prob_scores, test_prob_scores = get_scores_channelwise(
            distr_params=distributions_dic['univar_gaussian'],
            train_raw_scores=train_scores_tc,
            test_raw_scores=test_scores_tc,
            val_raw_scores=None,
            logcdf=True
        )

        score_t_test_dyn, score_tc_test_dyn, score_t_train_dyn, score_tc_train_dyn =  get_dynamic_scores(train_scores_tc, test_scores_tc, None,None, long_window=self.args.d_score_long_window, short_window=self.args.d_score_short_window)
        #input only scores with channels and time-points
        score_t_test_dyn_gauss_conv, score_tc_test_dyn_gauss_conv = get_gaussian_kernel_scores(None,score_tc_test_dyn,self.args.kernel_sigma)

        score_t_train_dyn_gauss_conv, score_tc_train_dyn_gauss_conv = get_gaussian_kernel_scores(None,score_tc_train_dyn,self.args.kernel_sigma)

        #test and train scores
        eval_metrics, anomaly_ratio = compute_metrics(test_ano_scores, gt, true_events,score_t_test_dyn, score_t_test_dyn_gauss_conv, self.args.seq_len , train_ano_scores, score_t_train_dyn, score_t_train_dyn_gauss_conv)
        #transform loss & memory usage into scalar
        avg_train_loss = np.mean(avg_train_loss)
        avg_test_loss= np.mean(test_metrics['loss'])

        total_allocated_memory_usage = np.concatenate([test_metrics['allocated_memory'],avg_allocated_memory_train])
        total_reserved_memory_usage = np.concatenate([test_metrics['reserved_memory'],avg_reserved_memory_train])

        total_allocated_memory_usage=np.mean(total_allocated_memory_usage)
        total_reserved_memory_usage=np.mean(total_reserved_memory_usage)

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
            batch_size=self.args.batch_size,
            anomaly_ratio=anomaly_ratio,
            embed=self.args.embed,
            total_time=total_time,
            train_duration=train_duration,
            test_duration=test_duration,
            metrics=eval_metrics,
            test_results_path=test_results_path,
            setting=setting,
            benchmark_id = self.args.benchmark_id,
            total_allocated_memory_usage=total_allocated_memory_usage,
            total_reserved_memory_usage=total_reserved_memory_usage
        )

        return