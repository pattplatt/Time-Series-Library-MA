export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=128
d_ff=128
e_layers=3
train_epochs=3

python -u run.py   --task_name classification   --is_training 1   --root_path ./dataset/anomaly_detection/Wafer/Wafer   --model_id wafer   --model iTransformer   --data UEA   --features S   --seq_len 50   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 1   --c_out 1   --top_k 3   --anomaly_ratio 1   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name classification   --is_training 1   --root_path ./dataset/anomaly_detection/Wafer/Wafer   --model_id wafer   --model Reformer   --data UEA   --features S   --seq_len 50   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 1   --c_out 1   --top_k 3   --anomaly_ratio 1   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name classification   --is_training 1   --root_path ./dataset/anomaly_detection/Wafer/Wafer   --model_id wafer   --model FEDformer   --data UEA   --features S   --seq_len 50   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 1   --c_out 1   --top_k 3   --anomaly_ratio 1   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name classification   --is_training 1   --root_path ./dataset/anomaly_detection/Wafer/Wafer   --model_id wafer   --model DLinear   --data UEA   --features S   --seq_len 50   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 1   --c_out 1   --top_k 3   --anomaly_ratio 1   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name classification   --is_training 1   --root_path ./dataset/anomaly_detection/Wafer/Wafer   --model_id wafer   --model Autoformer   --data UEA   --features S   --seq_len 50   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 1   --c_out 1   --top_k 3   --anomaly_ratio 1   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name classification   --is_training 1   --root_path ./dataset/anomaly_detection/Wafer/Wafer   --model_id wafer   --model Transformer   --data UEA   --features S   --seq_len 50   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 1   --c_out 1   --top_k 3   --anomaly_ratio 1   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name classification   --is_training 1   --root_path ./dataset/anomaly_detection/Wafer/Wafer   --model_id wafer   --model TimesNet   --data UEA   --features S   --seq_len 50   --pred_len 0   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 1   --c_out 1   --top_k 3   --anomaly_ratio 1   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name classification   --is_training 1   --root_path ./dataset/anomaly_detection/Wafer/Wafer   --model_id wafer   --model Informer   --data UEA   --features S   --seq_len 50   --pred_len 0   --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 1   --c_out 1   --top_k 3   --anomaly_ratio 1   --batch_size 128   --train_epochs $train_epochs --freq s
