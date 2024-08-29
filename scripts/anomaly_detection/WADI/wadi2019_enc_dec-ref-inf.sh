export CUDA_VISIBLE_DEVICES=0,1,2,3

e_layers=3
d_layers=3
train_epochs=3


python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler  --data_path WADI_train.csv   --model_id WADI_enc_dec   --model Reformer   --data WADI_F   --features M   --seq_len 96   --label_len 96   --pred_len 96   --e_layers $e_layers   --d_layers $d_layers   --factor 3   --enc_in 123   --dec_in 123   --c_out 123   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s --k_value 8

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler  --data_path WADI_train.csv   --model_id WADI_enc_dec   --model Informer   --data WADI_F   --features M   --seq_len 96   --label_len 96   --pred_len 96   --e_layers $e_layers   --d_layers $d_layers   --factor 3   --enc_in 123   --dec_in 123   --c_out 123   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s --k_value 8