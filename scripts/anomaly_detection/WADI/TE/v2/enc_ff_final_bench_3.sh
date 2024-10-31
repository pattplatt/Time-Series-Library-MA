export CUDA_VISIBLE_DEVICES=2

d_model=1024
d_ff=512
e_layers=5
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id FEDformer_enc_ff   --model FEDformer  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model $d_model   --d_ff $d_ff   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 16   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 256

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id TimesNet_enc_ff   --model TimesNet  --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 256   --d_ff 256   --e_layers $e_layers   --enc_in 123   --c_out 123   --batch_size 16   --train_epochs $train_epochs --freq s --benchmark_id enc_ff --dim_ff_dec 128






python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id TimesNet_enc_ff   --model TimesNet  --data WADI   --features M   --seq_len 30   --pred_len 0  --label_len 0  --d_model 256   --d_ff 256   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 16   --train_epochs 3 --freq s --benchmark_id enc_ff --dim_ff_dec 128