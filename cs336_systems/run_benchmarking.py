import pandas as pd
import subprocess
import benchmarking_script

# Parse the model configs into a list of dicts
class Config:
    def __init__(self, size: str, d_model: int, d_ff: int, num_layers: int, num_heads: int):
        self.size = size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads

configs = [
    Config("small", 768, 3072, 12, 12),
    Config("medium", 1024, 4096, 24, 16),
    Config("large", 1280, 5120, 36, 20),
    Config("xl", 1600, 6400, 48, 25),
    Config("2.7B", 2560, 10240, 32, 32)
]

context_lengths = [128, 256, 512, 1024]

# Run benchmarks and collect results
results = []
for context_length in context_lengths:
    print(f"\nBenchmarking {context_length} context length...")
    for config in configs:
        print(f"\nBenchmarking {config.size} model...")

        # Create command line arguments
        cmd_args = [
            "nsys", "profile",
            "-o", f"result_ctx{context_length}_{config.size}",
            "python", "benchmarking_script.py",
            "--device", "cuda",
            "--d_model", str(config.d_model),
            "--d_ff", str(config.d_ff),
            "--num_layers", str(config.num_layers),
            "--num_heads", str(config.num_heads),
            "--vocab_size", "10000",
            "--context_length", str(context_length),
            "--batch_size", "4",
            "--rope_theta", "1e6",
            "--warmup_steps", "5",
            "--n_steps", "10",
            "--run_backward"
        ]

        # Run the benchmark script with nsys profile
        process = subprocess.run(cmd_args, capture_output=True, text=True)

        # Parse the output to get timing results
        output_lines = process.stdout.split('\n')
        forward_time = None
        forward_stdev = None
        backward_time = None
        backward_stdev = None

        for line in output_lines:
            if "Forward pass:" in line:
                parts = line.split()
                forward_time = float(parts[2])
                forward_stdev = float(parts[4])
            elif "Backward pass:" in line:
                parts = line.split()
                backward_time = float(parts[2])
                backward_stdev = float(parts[4])

        results.append({
            'Model': config.size,
            'Mean Forward (ms)': forward_time,
            'Stdev Forward': forward_stdev,
            'Mean Backward': backward_time,
            'Stdev Backward': backward_stdev,
        })

    # Create DataFrame and format for LaTeX
    df = pd.DataFrame(results)
    latex_table = df.to_latex(index=False, float_format=lambda x: '{:.2f}'.format(x))
    print("\nLaTeX Table:")
    print(latex_table)
