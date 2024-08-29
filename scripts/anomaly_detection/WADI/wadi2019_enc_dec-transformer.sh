export CUDA_VISIBLE_DEVICES=0,1,2,3

e_layers=3
d_layers=3
train_epochs=3

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler  --data_path WADI_train.csv   --model_id WADI_enc_dec   --model Transformer   --data WADI_F   --features M   --seq_len 128   --label_len 96   --pred_len 128   --e_layers 3   --d_layers 3   --factor 3   --enc_in 123   --dec_in 123   --c_out 123   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs 3 --freq s --k_value 6.1

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler  --data_path WADI_train.csv   --model_id WADI_enc_dec   --model Transformer   --data WADI_F   --features M   --seq_len 128   --label_len 128   --pred_len 128   --e_layers 3   --d_layers 3   --factor 3   --enc_in 123   --dec_in 123   --c_out 123   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs 3 --freq s --k_value 6.1

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler  --data_path WADI_train.csv   --model_id WADI_enc_dec   --model Transformer   --data WADI_F   --features M   --seq_len 48   --label_len 48   --pred_len 48   --e_layers 3   --d_layers 3   --factor 3   --enc_in 123   --dec_in 123   --c_out 123   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs 3 --freq s --k_value 6.1

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler  --data_path WADI_train.csv   --model_id WADI_enc_dec   --model Transformer   --data WADI_F   --features M   --seq_len 256   --label_len 256   --pred_len 256   --e_layers 3   --d_layers 3   --factor 3   --enc_in 123   --dec_in 123   --c_out 123   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs 3 --freq s --k_value 6.1

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler  --data_path WADI_train.csv   --model_id WADI_enc_dec   --model Transformer   --data WADI_F   --features M   --seq_len 10   --label_len 10   --pred_len 10   --e_layers 3   --d_layers 3   --factor 3   --enc_in 123   --dec_in 123   --c_out 123   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs 3 --freq s --k_value 6.1

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler  --data_path WADI_train.csv   --model_id WADI_enc_dec   --model Transformer   --data WADI_F   --features M   --seq_len 70   --label_len 70   --pred_len 70   --e_layers 3   --d_layers 3   --factor 3   --enc_in 123   --dec_in 123   --c_out 123   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs 3 --freq s --k_value 6.1

