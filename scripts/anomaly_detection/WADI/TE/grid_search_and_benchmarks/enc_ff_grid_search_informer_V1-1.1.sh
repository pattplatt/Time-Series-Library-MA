export CUDA_VISIBLE_DEVICES=0,1,2,3

d_model=512
d_ff=256
e_layers=2
train_epochs=3

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_gs_seq_15_epochs   --model Informer   --data WADI   --features M   --seq_len 15   --pred_len 15  --label_len 15  --d_model 1024   --d_ff 512   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs 5 --freq s --n_heads 16

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_gs_seq_32   --model Informer   --data WADI   --features M   --seq_len 32   --pred_len 32  --label_len 32  --d_model 1024   --d_ff 512   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --n_heads 16

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_gs_seq_32_epochs   --model Informer   --data WADI   --features M   --seq_len 32   --pred_len 32  --label_len 32  --d_model 1024   --d_ff 512   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs 5 --freq s --n_heads 16

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_gs_seq_30   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 1024   --d_ff 512   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --n_heads 16

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_gs_seq_30_epochs   --model Informer   --data WADI   --features M   --seq_len 30   --pred_len 30  --label_len 30  --d_model 1024   --d_ff 512   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs 5 --freq s --n_heads 16

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_gs_seq_15_h32   --model Informer   --data WADI   --features M   --seq_len 15   --pred_len 15  --label_len 15  --d_model 1024   --d_ff 512   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --n_heads 32

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed   --model_id Informer_enc_ff_gs_seq_32_h32   --model Informer   --data WADI   --features M   --seq_len 32   --pred_len 32  --label_len 32  --d_model 1024   --d_ff 512   --e_layers 5   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs $train_epochs --freq s --n_heads 32