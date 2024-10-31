export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=1024
d_ff=512
e_layers=5
train_epochs=5

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_v2_dim256_relu   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_v2_dim256_dm512_dff1024_relu   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 512   --d_ff 1024   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_v2_dim256_dm256_dff1024_relu   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 256   --d_ff 1024   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_v2_dim256_dm128_dff1024_relu   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 128   --d_ff 1024   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_v2_dim256_dm256_dff2048_relu   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 256   --d_ff 2048   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_v2_dim256_dm64_dff512_relu   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 64   --d_ff 512   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256