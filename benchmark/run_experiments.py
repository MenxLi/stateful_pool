import subprocess
import time
import sys
import json
import os
import requests

# Configuration
SERVERS = [
    "exp/server_simple.py",
    "exp/server_mp.py",
    "exp/server_spool.py"
]
STRESS_SCRIPT = "exp/stress.py"
BATCH_SIZES = [1, 4, 8, 16, 32]
URL = "http://localhost:8000/predict"
RUNS_PER_SETUP = 5
NUM_REQUESTS = 100 
CONCURRENCY = 8

LOG_DIR = "exp_log"
RESULTS_FILE = os.path.join(LOG_DIR, "experiment_results.json")

def wait_for_server(url, timeout=60):
    start_time = time.time()
    base_url = "http://localhost:8000"
    while time.time() - start_time < timeout:
        try:
            requests.get(base_url, timeout=1) 
            return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    return False

def run_stress_test(batch_size):
    cmd = [
        sys.executable, STRESS_SCRIPT,
        "--url", URL,
        "--requests", str(NUM_REQUESTS),
        "--concurrency", str(CONCURRENCY),
        "--batch-size", str(batch_size)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        # Find the JSON block marked by delimiters
        start_marker = "JSON_START"
        end_marker = "JSON_END"
        start_idx = output.find(start_marker)
        end_idx = output.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            json_str = output[start_idx + len(start_marker):end_idx].strip()
            return json.loads(json_str)
        # Fallback for old behavior or if markers missing (though we just added them)
        idx = output.rfind('{')
        if idx != -1:
            json_str = output[idx:]
            return json.loads(json_str)
        return None
    except Exception as e:
        print(f"Stress test error: {e}")
        return None

def run_experiment(server_script):
    results = {}
    server_name = os.path.basename(server_script)
    log_file_path = os.path.join(LOG_DIR, f"{server_name}.log")
    
    env = os.environ.copy()
    root_dir = os.getcwd()
    env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")

    with open(log_file_path, "w") as log_file:
        print(f"Starting {server_name}...")
        server_process = subprocess.Popen(
            [sys.executable, server_script],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )

        try:
            if not wait_for_server(URL):
                print(f"Failed to start {server_name}")
                return None
            
            print(f"Server {server_name} ready.")
            
            for batch_size in BATCH_SIZES:
                print(f"  Batch size: {batch_size}")
                run_metrics = []
                for i in range(RUNS_PER_SETUP):
                    print(f"    Run {i+1}...", end="", flush=True)
                    metrics = run_stress_test(batch_size)
                    if metrics:
                        # Append the full metrics for this run
                        run_metrics.append(metrics)
                        print(f" Done. TPS: {metrics['image_throughput']:.1f}")
                    else:
                        print(" Failed.")
                
                # Store all runs for this batch size
                results[batch_size] = run_metrics
                    
        finally:
            print(f"Terminating {server_name}...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
            
            # Additional cleanup wait
            time.sleep(5)
            
    return results

def main():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    final_results = {}
    for server in SERVERS:
        if os.path.exists(server):
            res = run_experiment(server)
            if res:
                final_results[server] = res
        else:
            print(f"Server script {server} not found.")

    with open(RESULTS_FILE, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Done. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
