import subprocess

# Define the hyperparameter values to try
d_ff_values = [64, 128, 256,512]
e_layers_values = [2, 3, 5]
d_model_values=[64, 128, 256,512]
batch_size_values=[32,64,128,256,512]
learning_rate_values=[0.01,0.001,0.0001,0.00001]

# Base command and arguments
base_command = [
    "python", "-u", "run.py",
    "--task_name", "anomaly_detection",
    "--is_training", "1",
    "--root_path", "./dataset/anomaly_detection/WADI/WADI2019_no_scaler",
    "--model_id", "WADI",
    "--model", "iTransformer",
    "--data", "WADI",
    "--features", "M",
    "--seq_len", "10",
    "--pred_len", "0",
    "--enc_in", "123",
    "--c_out", "123",
    "--top_k", "3",
    "--anomaly_ratio", "0.5",
    "--train_epochs", "3",
    "--freq", "s"
]

# Loop over hyperparameter combinations
for d_ff in d_ff_values:
    for e_layers in e_layers_values:
        for d_model in d_model_values:
            for batch_size in batch_size_values:
                for learning_rate in learning_rate_values:
                    # Construct the command with current hyperparameters
                    command = base_command + [
                        "--d_ff", str(d_ff),
                        "--e_layers", str(e_layers),
                        "--d_model", str(d_model),
                        "--batch_size", str(batch_size),
                        "--learning_rate", str(learning_rate)
                    ]

                    # Print the command for reference
                    print(f"Running command: {' '.join(command)}")

                    # Run the command
                    result = subprocess.run(command, capture_output=True, text=True)

                    # Print the output and error (if any) of the command
                    print(result.stdout)
                    if result.stderr:
                        print(f"Error: {result.stderr}")
