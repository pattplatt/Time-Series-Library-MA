export CUDA_VISIBLE_DEVICES=0,1,2,3

e_layers=3
train_epochs=3

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Transformer_UAE_grid_search_v1-0  --model Transformer   --data WADI   --features S   --seq_len 15   --pred_len 15 --label_len 15   --d_model 32   --d_ff 32   --e_layers 2   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 3  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Transformer_UAE_grid_search_v1-0_enc5  --model Transformer   --data WADI   --features S   --seq_len 15   --pred_len 15 --label_len 15   --d_model 32   --d_ff 32   --e_layers 5   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 3  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Transformer_UAE_grid_search_v1-0_enc5_heads16  --model Transformer   --data WADI   --features S   --seq_len 15   --pred_len 15 --label_len 15   --d_model 32   --d_ff 32   --e_layers 5   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 3  --num_workers 0 --n_heads 16

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Transformer_UAE_grid_search_v1-0_dimSmall_enc5  --model Transformer   --data WADI   --features S   --seq_len 15   --pred_len 15 --label_len 15   --d_model 16   --d_ff 16   --e_layers 5   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 3  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Transformer_UAE_grid_search_v1-0_dimBig_enc5  --model Transformer   --data WADI   --features S   --seq_len 15   --pred_len 15 --label_len 15   --d_model 64   --d_ff 64   --e_layers 5   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 3  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Transformer_UAE_grid_search_v1-0_dimBig_enc5_seq30  --model Transformer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 64   --d_ff 64   --e_layers 5   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 3  --num_workers 0 --n_heads 8

python -u run.py   --task_name anomaly_detection_uae   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/WADI2019_power_transformed  --model_id Transformer_UAE_grid_search_v1-0_dimBig_enc5_seq15  --model Transformer   --data WADI   --features S   --seq_len 15   --pred_len 15 --label_len 15   --d_model 64   --d_ff 64   --e_layers 5   --enc_in 123  --c_out 1   --batch_size 128   --train_epochs 3  --num_workers 0 --n_heads 8
