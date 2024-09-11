from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjustment, get_gaussian_kernel_scores, get_dynamic_scores, threshold_and_predict
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
import csv

warnings.filterwarnings('ignore')

class Exp_Anomaly_Enc_Dec(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Enc_Dec, self).__init__(args)

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
                    #total_outputs.append(outputs)
                
                score = torch.mean(outputs, dim=-1) 
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                #total_outputs_tensor=torch.stack(total_outputs, dim=0)
                
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
            #average of train loss
            k = self.args.k_value
            anomaly_threshold = np.average(train_loss) + (np.std(train_loss)) + k
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
            
        #total_outputs_tensor = total_outputs_tensor.view(total_outputs_tensor.shape[0]*total_outputs_tensor.shape[1], 48, 123)
        #total_outputs_tensor=total_outputs_tensor.cpu().detach().numpy() 
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        train_end_time = time.time()
        train_duration = train_end_time - training_start_time

        print(f"Train_loss: {train_loss}")
        print(f"anomaly_threshold: {anomaly_threshold}")
        
        return self.model, train_loss, vali_loss, test_loss, train_duration, anomaly_threshold, k
    
    #"###########################################"
    
    #"###########################################"
    
    #"###########################################"

    def test(self, anomaly_threshold, setting,train_loss, vali_loss, test_loss,model,seq_len ,d_model,e_layers,d_ff,n_heads,train_epochs,loss,learning_rate,anomaly_ratio,embed,train_duration, k, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        #print("test_data.__len__",test_data.__len__())
        #batch_x, batch_y, batch_x_mark, batch_y_mark=test_data.__getitem__(0)
        #print("batch_x:",batch_x.shape)
        #print("batch_y:",batch_y.shape)
        #batch_x, batch_y, batch_x_mark, batch_y_mark=test_data.__getitem__(99)
        
        test_start_time = time.time()
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        criterion = self._select_criterion()
        preds = []
        trues = []
        anomaly_preds=[]
        labels=[]
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        total_losses = []
        
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
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                #print("outputs.shape:",outputs.shape,"batch_y.shape",batch_y.shape)

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                #print("outputs.shape",outputs.shape)
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                labels.append(seq_labels)
                #print(f"preds.shape:{np.array(preds).shape}")
                #print(f"outputs.shape:{outputs.shape}")
                labels_ = torch.stack(labels)
                #print(f"labels.shape:{labels_.shape}")
                
            preds=np.array(preds)
                
            for i in range(preds.shape[0]):
                for n in range(preds.shape[1]):
                    for l in range(preds.shape[2]):
                    
                        batch = preds[i][n][l]
                        #print(f"batch.shape:{batch.shape}")
                        #das hier sind die MSE errors als anomaly scores
                        l1_loss = nn.L1Loss()

                        loss = criterion(torch.from_numpy(batch), torch.from_numpy(batch_y[l]))
                        #print(f"loss: {loss}")
                        #print(f"anomaly_threshold: {anomaly_threshold}")
                        total_losses.append(loss)
                    
                    #das raus, anomaly_classification wird erst zum schluss mit allen punkten gemacht
                    #anomaly_classification = loss > anomaly_threshold
                    #anomaly_classification_array = np.full(pred.shape[1], anomaly_classification)
       #print("anomaly_classification_array.shape",np.array(anomaly_classification_array).shape)
                    #print("anomaly_classification_array.flatten.shape",np.array(anomaly_classification_array).flatten().shape)
                    #anomaly_preds.append(anomaly_classification_array)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        total_losses = np.array(total_losses)
        print(f"total_losses.shape:{total_losses.shape}")
        
        #this should be a run parameter 
        kernel_sigma = 3 
        #input only scores with channels and time-points
        #score_t_test, score_tc_test = get_gaussian_kernel_scores(total_losses,None,kernel_sigma)
        score_t_test, _, _, _ =  get_dynamic_scores(None, None, None, total_losses, long_window=100000, short_window=100)
        
        preds = np.array(preds)
        trues = np.array(trues)
        #print('preds shape:', preds.shape)
        #print('trues shape:', trues.shape)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('test shape:', preds.shape, trues.shape)
        print("trues:",trues.shape)
        
        anomaly_preds = torch.tensor(anomaly_preds).flatten()
        
        # result save
        folder_path = './test_results/' + setting + '/'
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
        
        test_end_time = time.time()
        test_duration = test_end_time - test_start_time
        total_time = train_duration + test_duration
        
        labels = torch.stack(labels, dim=0)
        labels = labels.flatten()
        
        print(f"labels.shape:{labels.shape}")
        
        opt_thres, pred_labels, avg_prec, auroc = threshold_and_predict(score_t_test,labels,true_events=None,logger=None,test_anom_frac=0.1,thres_method="tail_prob",return_auc=True,)
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f_score, support = precision_recall_fscore_support(labels, pred_labels, average='binary')
        cm = confusion_matrix(labels, pred_labels)
        
        #(4) detection adjustment
        #if self.args.point_adjustment:
        labels_point_adjusted, anomaly_preds_point_adjusted = adjustment(labels, pred_labels)
        accuracy_point_adjusted = accuracy_score(labels_point_adjusted, anomaly_preds_point_adjusted)
        precision_point_adjusted, recall_point_adjusted, f_score_point_adjusted, support_point_adjusted = precision_recall_fscore_support(labels_point_adjusted, anomaly_preds_point_adjusted, average='binary')
        cm_point_adjusted = confusion_matrix(labels_point_adjusted, anomaly_preds_point_adjusted)
        
        #print("labels.shape",labels.shape)
        #print("anomaly_preds.shape",anomaly_preds.shape)
        
        print("accuracy:",accuracy)
        print("accuracy_point_adjusted:",accuracy_point_adjusted)
        print("Anomaly Threshold:",anomaly_threshold)
        print("test loss average:",np.average(total_losses), "test loss min:", np.min(total_losses), "test loss max:", np.max(total_losses))
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        print("Accuracy_point_adjusted : {:0.4f}, Precision_point_adjusted : {:0.4f}, Recall_point_adjusted : {:0.4f}, F-score_point_adjusted : {:0.4f} ".format(
            accuracy_point_adjusted, precision_point_adjusted,
            recall_point_adjusted, f_score_point_adjusted))
        print("cm")
        print(cm)
        print("cm_point_adjusted")
        print(cm_point_adjusted)
        #print("total number of predictions: ", len(labels)

        parameters_header = ["Model", "k_value", "Train loss", "Vali loss", "Test loss", "Seq len", "Dim Model", "E layers", "Dim ff", "n_heads", "Train epochs", "Loss", "Learning rate", "Anomaly ratio", "embed", "Total Duration (s)", "Train Duration (s)", "Test Duration (s)"]
        parameters = [model, k, f"{train_loss:.6f}", f"{vali_loss:.6f}", f"{np.average(total_losses):.6f}", seq_len, d_model, e_layers, d_ff, n_heads, train_epochs, criterion, learning_rate, anomaly_ratio, embed,f"{total_time:.2f}", f"{train_duration:.2f}", f"{test_duration:.2f}"]
        metrics_f1_header = ["Model", "Accuracy", "Precision", "Recall", "F-score", "Confusion Matrix"]
        metrics_f1 = [model, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f_score:.4f}", f"{cm}"]
        metrics_fpa_header = ["Model", "Accuracy_fpa", "Precision_fpa", "Recall_fpa", "F-score_fpa", "Confusion Matrix_fpa"]
        metrics_fpa = [model, f"{accuracy_point_adjusted:.4f}", f"{precision_point_adjusted:.4f}", f"{recall_point_adjusted:.4f}", f"{f_score_point_adjusted:.4f}", f"{cm_point_adjusted}"]

        file_path = os.path.join(folder_path, "results_anomaly_enc_dec_slide_test.csv")
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(parameters_header)
            writer.writerow(parameters)
            writer.writerow([""])
            writer.writerow(metrics_f1_header)
            writer.writerow(metrics_f1)
            writer.writerow([""])
            writer.writerow(metrics_fpa_header)
            writer.writerow(metrics_fpa)

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
