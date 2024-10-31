export CUDA_VISIBLE_DEVICES=0,1,2,3

e_layers=2
d_layers=2
train_epochs=1
l_w=100000
s_w=1

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_AutoformerV1.1  --model Autoformer   --data WADI_F   --features M   --seq_len 30   --label_len 30   --pred_len 30 --n_heads 8 --d_model 512 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 256 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'slide'

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_TransformerV1.1   --model Transformer   --data WADI_F   --features M   --seq_len 30   --label_len 30   --pred_len 30 --n_heads 8 --d_model 512 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 256 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'slide' 

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_iTransformerV1.1   --model iTransformer   --data WADI_F   --features M   --seq_len 30   --label_len 30   --pred_len 30 --n_heads 8 --d_model 512 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 256 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'slide' 

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_FEDformerV1.1 --model FEDformer   --data WADI_F   --features M   --seq_len 30   --label_len 30   --pred_len 30 --n_heads 8 --d_model 512 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 256 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'slide'

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_DLinearV1.1   --model DLinear   --data WADI_F   --features M   --seq_len 30   --label_len 30   --pred_len 30 --n_heads 8 --d_model 512 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 256 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'slide'

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_TimesNetV1.1  --model TimesNet   --data WADI_F   --features M   --seq_len 30   --label_len 30   --pred_len 30 --n_heads 8 --d_model 64 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 32 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'slide'

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_InformerV1.1   --model Informer   --data WADI_F   --features M   --seq_len 30   --label_len 30   --pred_len 30 --n_heads 8 --d_model 512 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 256 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'slide'

python -u run.py   --task_name enc_dec_anomaly   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --data_path WADI_train.csv   --model_id WADI_enc_dec_ReformerV1.1   --model Reformer   --data WADI_F   --features M   --seq_len 30   --label_len 30   --pred_len 30 --n_heads 8 --d_model 512 --enc_in 123 --e_layers $e_layers --dec_in 123  --d_layers $d_layers --c_out 123 --d_ff 256 --factor 1   --des 'Exp'   --itr 1 --batch_size 128 --train_epochs $train_epochs --freq s  --win_mode 'slide'
