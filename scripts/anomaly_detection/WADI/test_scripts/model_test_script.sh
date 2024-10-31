export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u run.py   --task_name enc_dec_anomaly  --is_training 1   --root_path ./dataset/anomaly_detection/WADI/mini_wadi   --model_id inf_test_enc_dec_1   --model Informer   --data WADI_F   --features M   --seq_len 30   --pred_len 30 --label_len 30   --d_model 256   --d_ff 128   --e_layers 1   --enc_in 123   --c_out 123  --batch_size 128   --train_epochs 1 --freq s --dec_in 123 --benchmark_id bench_1

python -u run.py   --task_name anomaly_detection_uae  --is_training 1   --root_path ./dataset/anomaly_detection/WADI/mini_wadi   --model_id inf_test_uae_1   --model Informer   --data WADI   --features S   --seq_len 30   --pred_len 30 --label_len 30   --d_model 32   --d_ff 32   --e_layers 1   --enc_in 10   --c_out 1    --batch_size 64   --train_epochs 1 --freq s --num_workers 0 --benchmark_id bench_2

python -u run.py   --task_name anomaly_detection   --is_training 1   --root_path ./dataset/anomaly_detection/WADI/mini_wadi   --model_id inf_test_enc_ff_1    --model Informer --data WADI   --features M   --seq_len 30   --pred_len 30 --label_len 30   --d_model 256   --d_ff 128   --e_layers 1   --enc_in 123   --c_out 123   --batch_size 128   --train_epochs 1 --freq s --benchmark_id bench_3
