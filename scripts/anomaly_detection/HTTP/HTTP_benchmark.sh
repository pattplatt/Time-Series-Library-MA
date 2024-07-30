export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=128
d_ff=128
e_layers=3
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/HTTP   --model_id HTTP   --model iTransformer   --data HTTP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 3   --c_out 3   --top_k 3   --anomaly_ratio 0.8   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/HTTP   --model_id HTTP   --model Reformer   --data HTTP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 3   --c_out 3   --top_k 3   --anomaly_ratio 0.8   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/HTTP   --model_id HTTP   --model FEDformer   --data HTTP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 3   --c_out 3   --top_k 3   --anomaly_ratio 0.8   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/HTTP   --model_id HTTP   --model DLinear   --data HTTP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 3   --c_out 3   --top_k 3   --anomaly_ratio 0.8   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/HTTP   --model_id HTTP   --model Autoformer   --data HTTP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 3   --c_out 3   --top_k 3   --anomaly_ratio 0.8   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/HTTP   --model_id HTTP   --model Transformer   --data HTTP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 3   --c_out 3   --top_k 3   --anomaly_ratio 0.8   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/HTTP   --model_id HTTP   --model TimesNet   --data HTTP   --features M   --seq_len 100   --pred_len 0   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 3   --c_out 3   --top_k 3   --anomaly_ratio 0.8   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/HTTP   --model_id HTTP   --model Informer   --data HTTP   --features M   --seq_len 100   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 3   --c_out 3   --top_k 3   --anomaly_ratio 0.8   --batch_size 128   --train_epochs $train_epochs --freq s
