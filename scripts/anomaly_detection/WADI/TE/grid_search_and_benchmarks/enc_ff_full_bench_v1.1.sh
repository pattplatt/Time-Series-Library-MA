export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=512
d_ff=256
e_layers=2
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Transformer_enc_ff_v1.1   --model Transformer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Autoformer_enc_ff_v1.1   --model Autoformer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_v1.1   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id iTransformer_enc_ff_v1.1   --model iTransformer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id FEDformer_enc_ff_v1.1   --model FEDformer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id TimesNet_enc_ff_v1.1   --model TimesNet   --data WADI   --features M   --seq_len 30   --pred_len 0  --label_len 30  --d_model 64   --d_ff 32   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Reformer_enc_ff_v1.1   --model Reformer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id DLinear_enc_ff_v1.1   --model DLinear   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s
