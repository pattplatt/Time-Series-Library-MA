export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=128
d_ff=128
e_layers=3
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler   --model_id WADI   --model Autoformer   --data WADI   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --top_k 3   --anomaly_ratio 0.5   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler   --model_id WADI   --model Transformer   --data WADI   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --top_k 3   --anomaly_ratio 0.5   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler   --model_id WADI   --model TimesNet   --data WADI   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --top_k 3   --anomaly_ratio 0.5   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_no_scaler   --model_id WADI   --model Informer   --data WADI   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --top_k 3   --anomaly_ratio 0.5   --batch_size 128   --train_epochs $train_epochs --freq s