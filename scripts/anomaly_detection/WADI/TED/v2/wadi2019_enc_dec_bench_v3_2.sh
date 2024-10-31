export CUDA_VISIBLE_DEVICES=0

e_layers=5
d_layers=5
train_epochs=5
l_w=100000
s_w=1

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_TransformerV3  --model Transformer   --data WADI_F   --features M   --seq_len 5   --label_len 5   --pred_len 5 --n_heads 8 --d_model 256 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 1024    --des 'Exp'    --batch_size 64 --train_epochs $train_epochs --freq s  --win_mode 'hopping' --benchmark_id enc_dec

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_AutoformerV3   --model Autoformer   --data WADI_F   --features M  --seq_len 5   --label_len 5   --pred_len 5 --n_heads 8 --d_model 256 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 1024    --des 'Exp'    --batch_size 64 --train_epochs $train_epochs --freq s  --win_mode 'hopping' --benchmark_id enc_dec

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_InformerV3   --model Informer   --data WADI_F   --features M   --seq_len 5   --label_len 5   --pred_len 5 --n_heads 8 --d_model 256 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 1024    --des 'Exp'    --batch_size 64 --train_epochs $train_epochs --freq s  --win_mode 'hopping' --benchmark_id enc_dec

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_iTransformerV3 --model iTransformer   --data WADI_F   --features M   --seq_len 5   --label_len 5   --pred_len 5 --n_heads 8 --d_model 256 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 1024    --des 'Exp'    --batch_size 64 --train_epochs $train_epochs --freq s  --win_mode 'hopping' --benchmark_id enc_dec

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_ReformerV3   --model Reformer   --data WADI_F   --features M   --seq_len 5   --label_len 5   --pred_len 5 --n_heads 8 --d_model 256 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 1024    --des 'Exp'    --batch_size 64 --train_epochs $train_epochs --freq s  --win_mode 'hopping' --benchmark_id enc_dec

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_DLinearV3   --model DLinear   --data WADI_F   --features M   --seq_len 5   --label_len 5   --pred_len 5 --n_heads 8 --d_model 256 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 1024    --des 'Exp'    --batch_size 64 --train_epochs $train_epochs --freq s  --win_mode 'hopping' --benchmark_id enc_dec