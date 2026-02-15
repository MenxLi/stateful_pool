import time
import argparse
import concurrent.futures
import requests
import numpy as np
from PIL import Image
import json
import io
import os

def create_random_image_bytes():
    # Random uni-color
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    arr = np.full((224, 224, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def send_request_task(url, batch_size):
    try:
        image_data = create_random_image_bytes()
        
        start_time = time.time()
        # We reuse the same image data for all files in the batch for maximum throughput
        files = [('files', (f'random_{i}.png', image_data, 'image/png')) for i in range(batch_size)]
        
        response = requests.post(url, files=files)
        latency = time.time() - start_time
        return response.status_code, latency, None
    except Exception as e:
        return 0, 0, str(e)

def stress_test(url, num_requests, concurrency, batch_size):
    print(f"Starting stress test on {url}")
    
    max_workers = max(2, os.cpu_count() // 4)   # type: ignore
    actual_workers = min(concurrency, max_workers)
    
    print(f"Total requests: {num_requests}, Max Allowed Concurrency: {max_workers}, Actual Concurrency: {actual_workers}, Batch Size: {batch_size}")
    
    start_time = time.time()
    latencies = []
    success_count = 0
    error_count = 0
    errors = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
        # Submit tasks
        futures = [executor.submit(send_request_task, url, batch_size) for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            status, latency, error_msg = future.result()
            if status == 200:
                success_count += 1
                latencies.append(latency)
            else:
                error_count += 1
                if error_msg:
                    errors[error_msg] = errors.get(error_msg, 0) + 1

    total_time = time.time() - start_time

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = (success_count * batch_size) / total_time if total_time > 0 else 0 # Images per second
    req_throughput = success_count / total_time if total_time > 0 else 0 # Requests per second

    result = {
        "total_time": total_time,
        "successful_requests": success_count,
        "failed_requests": error_count,
        "request_throughput": req_throughput,
        "image_throughput": throughput,
        "average_latency": avg_latency,
        "error_details": errors
    }
    print("JSON_START")
    print(json.dumps(result, indent=4))
    print("JSON_END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test the inference server.")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="Server URL")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=16, help="Number of concurrent requests")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per request")
    
    args = parser.parse_args()
    
    stress_test(args.url, args.requests, args.concurrency, args.batch_size)
