from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        attens_energy = []
        total_outputs=[]

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        training_start_time=time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    #print("train: outputs.shape",outputs.shape)
                    total_outputs.append(outputs)
                
                score = torch.mean(outputs, dim=-1) 
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                total_outputs_tensor=torch.stack(total_outputs, dim=0)
                #print("total_outputs_tensor", total_outputs_tensor.shape)
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        total_outputs_tensor = total_outputs_tensor.view(total_outputs_tensor.shape[0]*total_outputs_tensor.shape[1], 48, 123)
        print("flattened_tensor",total_outputs_tensor.shape)
        total_outputs_tensor=total_outputs_tensor.cpu().detach().numpy() 
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        train_end_time = time.time()
        train_duration = train_end_time - training_start_time
        
        # Calculate the mean along the second dimension (axis=1)
        mean_values = np.mean(total_outputs_tensor, axis=(0,2))
        std_values = np.std(total_outputs_tensor, axis=(0,2))
        k = 30  # This can be any value depending on how strict or loose you want the threshold
        anomaly_thresholds = (mean_values + k * std_values, mean_values - k * std_values)
        
        print("std_values",std_values.shape)
        print("mean_values:",mean_values.shape)
        
        #print("\nAnomaly Thresholds (mean ± k*std):")
        print(f"Upper Threshold: {anomaly_thresholds[0]}")
        print(f"Upper Threshold: {anomaly_thresholds[0]}")
        
        print(f"Lower Threshold: {anomaly_thresholds[1]}")

        return self.model, train_loss, vali_loss, test_loss, train_duration, anomaly_thresholds

    def test(self, anomaly_thresholds, setting,train_loss, vali_loss, test_loss,model,seq_len ,d_model,e_layers,d_ff,n_heads,train_epochs,loss,learning_rate,anomaly_ratio,embed,train_duration, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        preds = []
        trues = []
        anomaly_preds=[]
        labels=[]
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,seq_labels) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                #print("outputs.shape:",outputs.shape)
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                labels.append(seq_labels)
                preds_mean=np.mean(pred, axis=2, keepdims=True)
                
                print("preds_mean.shape",preds_mean.shape)                
                print("preds_mean",np.mean(preds_mean,axis=1))
                upper_anomaly_thresholds = anomaly_thresholds[0].reshape(-1,1)
                lower_anomaly_thresholds = anomaly_thresholds[1].reshape(-1,1)
                print("before broadcast lower_anomaly_thresholds.shape",lower_anomaly_thresholds.shape)
                #lower_anomaly_thresholds = np.broadcast_to(preds_mean, (128, 48, 1))
                print("after broadcast lower_anomaly_thresholds.shape",lower_anomaly_thresholds.shape)

                
                anomaly_classification = preds_mean < lower_anomaly_thresholds
                anomaly_classification_2 = preds_mean == lower_anomaly_thresholds
                
                num_equal = np.sum(anomaly_classification_2)
                num_true = np.sum(anomaly_classification)
                num_false = anomaly_classification.size - num_true

                # Print the results
                print("Number of equal values:", num_equal)
                
                print("Number of True values:", num_true)
                print("Number of False values:", num_false)
                #print("anomaly_classification",anomaly_classification.shape)
                #print("seq_labels",seq_labels.shape)
                
                anomaly_preds.append(anomaly_classification)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        #upper_anomaly_thresholds = anomaly_thresholds[1].reshape(-1,1)

        print("anomaly_classification.shape",np.array(anomaly_classification).shape)
        print("anomaly_classification.flatten.shape",np.array(anomaly_classification).flatten().shape)
        print("trues:",trues.shape)
        
        anomaly_preds = torch.tensor(anomaly_preds).flatten()
        print("lower_anomaly_thresholds",lower_anomaly_thresholds)
        print("upper_anomaly_thresholds",upper_anomaly_thresholds)
        
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
        
        labels = torch.stack(labels, dim=0)
        labels = labels.flatten()
        
        print("labels",labels.shape)

        accuracy = accuracy_score(labels, anomaly_preds)
        print("accuracy",accuracy)
        precision, recall, f_score, support = precision_recall_fscore_support(labels, anomaly_preds, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        cm = confusion_matrix(labels, anomaly_preds)
        #cm = np.floor(cm/seq_len)
        print("")
        print(cm)
        
        #mae, mse, rmse, mape, mspe = metric(anomaly_pred, random_bools)
        #print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        #f = open("result_long_term_forecast.txt", 'a')
        #f.write(setting + "  \n")
        #f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        #f.write('\n')
        #f.write('\n')
        #f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)

        return