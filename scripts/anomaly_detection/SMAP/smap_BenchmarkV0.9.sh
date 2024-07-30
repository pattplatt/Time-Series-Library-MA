export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=128
d_ff=128
e_layers=3
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SMAP   --model_id SMAP   --model iTransformer   --data SMAP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 25   --c_out 25   --top_k 3   --anomaly_ratio 9.72   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SMAP   --model_id SMAP   --model Reformer   --data SMAP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 25   --c_out 25   --top_k 3   --anomaly_ratio 9.72   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SMAP   --model_id SMAP   --model FEDformer   --data SMAP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 25   --c_out 25   --top_k 3   --anomaly_ratio 9.72   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SMAP   --model_id SMAP   --model DLinear   --data SMAP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 25   --c_out 25   --top_k 3   --anomaly_ratio 9.72   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SMAP   --model_id SMAP   --model Autoformer   --data SMAP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 25   --c_out 25   --top_k 3   --anomaly_ratio 9.72   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SMAP   --model_id SMAP   --model Transformer   --data SMAP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 25   --c_out 25   --top_k 3   --anomaly_ratio 9.72   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SMAP   --model_id SMAP   --model TimesNet   --data SMAP   --features M   --seq_len 100   --pred_len 0   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 25   --c_out 25   --top_k 3   --anomaly_ratio 9.72   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/SMAP   --model_id SMAP   --model Informer   --data SMAP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 25   --c_out 25   --top_k 3   --anomaly_ratio 9.72   --batch_size 128   --train_epochs $train_epochs --freq s
