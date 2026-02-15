import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

RESULTS_FILE = "exp_log/experiment_results.json"
OUTPUT_FILE = "exp_log/results_plot.png"

def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please run experiments first.")
        return

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    # Transform JSON to DataFrame
    records = []
    for server_path, batches in data.items():
        server_name = os.path.basename(server_path).replace(".py", "")
        
        # Mapping names for better readability in plot
        name_map = {
            "server_simple": "random dispatch (threaded)",
            "server_mp": "random dispatch (process)",
            "server_spool": "stateful-pool*"
        }
        display_name = name_map.get(server_name, server_name)

        for batch_size, runs in batches.items():
            if not runs: continue
            for run in runs:
                records.append({
                    "Server": display_name,
                    "Batch Size": int(batch_size),
                    "Image Throughput (img/s)": run["image_throughput"],
                    "Latency (s)": run["average_latency"]
                })
    
    if not records:
        print("No data found to visualize.")
        return

    df = pd.DataFrame(records)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Throughput
    sns.lineplot(
        data=df, 
        x="Batch Size", 
        y="Image Throughput (img/s)", 
        hue="Server", 
        style="Server",
        markers=True, 
        dashes=False, 
        err_style="band", # This adds the std/ci band
        errorbar="sd",    # requested "std range"
        ax=axes[0]
    )
    axes[0].set_title("Image Throughput vs Payload")
    axes[0].set_ylabel("Througput (img/s)")
    axes[0].set_xticks(df["Batch Size"].unique())

    # Plot Latency
    sns.lineplot(
        data=df, 
        x="Batch Size", 
        y="Latency (s)", 
        hue="Server", 
        style="Server",
        markers=True, 
        dashes=False, 
        err_style="band",
        errorbar="sd",
        ax=axes[1]
    )
    axes[1].set_title("Average Latency vs Payload")
    axes[1].set_ylabel("Latency (s)")
    axes[1].set_xticks(df["Batch Size"].unique())
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE)
    print(f"Plot saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
