export CUDA_VISIBLE_DEVICES=1

e_layers=5
d_layers=5
train_epochs=5
l_w=100000
s_w=1

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_FEDFormerV3   --model FEDformer   --data WADI_F   --features M   --seq_len 5   --label_len 5   --pred_len 5 --n_heads 8 --d_model 256 --enc_in 123 --e_layers 5 --dec_in 123  --d_layers 5 --c_out 123 --d_ff 1024    --des 'Exp'    --batch_size 64 --train_epochs 5 --freq s  --win_mode 'hopping' --benchmark_id enc_dec

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_TimesNetV3  --model TimesNet   --data WADI_F   --features M   --seq_len 5   --label_len 0   --pred_len 0 --n_heads 8 --d_model 64 --enc_in 123 --e_layers 5 --dec_in 123  --d_layers 5 --c_out 123 --d_ff 128    --des 'Exp'    --batch_size 64 --train_epochs 5 --freq s  --win_mode 'hopping' --benchmark_id enc_dec