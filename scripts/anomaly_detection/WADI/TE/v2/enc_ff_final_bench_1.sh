export CUDA_VISIBLE_DEVICES=1

d_model=1024
d_ff=512
e_layers=5
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Transformer_enc_ff   --model Transformer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 16   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Autoformer_enc_ff   --model Autoformer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 16   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256