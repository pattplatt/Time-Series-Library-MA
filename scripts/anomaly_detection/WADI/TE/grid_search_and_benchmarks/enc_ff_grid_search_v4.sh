export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=1024
d_ff=512
e_layers=5
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_grid_s_bs_1024   --model Informer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 1024   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_grid_s_bs_512   --model Informer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 512   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_grid_s_bs_256   --model Informer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 256   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_grid_s_bs_128   --model Informer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_grid_s_bs_64   --model Informer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 64   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_grid_s_bs_32   --model Informer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 32   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_grid_s_bs_16   --model Informer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 1024   --d_ff 512   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 16   --train_epochs 3 --freq s --benchmark_id enc_ff --dim_ff_dec 256

