export CUDA_VISIBLE_DEVICES=0,1,2,3

e_layers=3
d_layers=3
train_epochs=3

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Transformer_UAE_full_v1.0  --model Transformer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 1  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Autoformer_UAE_full_v1.0  --model Autoformer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 1  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Informer_UAE_full_v1.0  --model Informer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 3  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id iTransformer_UAE_full_v1.0  --model iTransformer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 1  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id FEDformer_UAE_full_v1.0  --model FEDformer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 1  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id TimesNet_UAE_full_v1.0  --model TimesNet   --data WADI   --features S   --seq_len 30   --pred_len 0 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 1  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Reformer_UAE_full_v1.0  --model Reformer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 1  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id DLinear_UAE_full_v1.0  --model DLinear   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 1  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id TFT_UAE_full_v1.1  --model TemporalFusionTransformer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 1  --num_workers 0 --n_heads 8