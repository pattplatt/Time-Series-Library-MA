export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=128
d_ff=128
e_layers=3
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SWaT   --model_id SWAT   --model iTransformer   --data SWAT   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 51   --c_out 51   --top_k 3   --anomaly_ratio 2.55   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SWaT   --model_id SWAT   --model Reformer   --data SWAT   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 51   --c_out 51   --top_k 3   --anomaly_ratio 2.55   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SWaT   --model_id SWAT   --model FEDformer   --data SWAT   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 51   --c_out 51   --top_k 3   --anomaly_ratio 2.55   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SWaT   --model_id SWAT   --model DLinear   --data SWAT   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 51   --c_out 51   --top_k 3   --anomaly_ratio 2.55   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SWaT   --model_id SWAT   --model Autoformer   --data SWAT   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 51   --c_out 51   --top_k 3   --anomaly_ratio 2.55   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SWaT   --model_id SWAT   --model Transformer   --data SWAT   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 51   --c_out 51   --top_k 3   --anomaly_ratio 2.55   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SWaT   --model_id SWAT   --model TimesNet   --data SWAT   --features M   --seq_len 100   --pred_len 0   --d_model 64   --d_ff 64   --e_layers 2   --enc_in 51   --c_out 51   --top_k 3   --anomaly_ratio 2.55   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SWaT   --model_id SWAT   --model Informer   --data SWAT   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 51   --c_out 51   --top_k 3   --anomaly_ratio 2.55   --batch_size 128   --train_epochs $train_epochs --freq s