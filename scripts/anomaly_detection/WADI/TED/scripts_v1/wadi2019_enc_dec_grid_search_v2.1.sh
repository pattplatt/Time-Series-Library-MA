export CUDA_VISIBLE_DEVICES=0,1,2,3

e_layers=3
d_layers=3
train_epochs=3

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_Reformer_gs_hopp_s5_dff1024_dm256_p15  --model Reformer   --data WADI_F   --features M --seq_len 50 --label_len 50 --pred_len 50 --n_heads 8 --d_model 256 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 1024 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'hopping'  --benchmark_id enc_dec

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_Reformer_gs_hopp_s5_dff1024_dm256_p30  --model Reformer   --data WADI_F   --features M --seq_len 96 --label_len 96 --pred_len 96 --n_heads 8 --d_model 256 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 1024 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'hopping'  --benchmark_id enc_dec

