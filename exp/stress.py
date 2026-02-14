import time
import argparse
import concurrent.futures
import requests
import numpy as np
from PIL import Image
import json
import io

def generate_random_image():
    # Generate a random 224x224 RGB image
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

def send_request(url, image_data, batch_size=1):
    try:
        start_time = time.time()
        files = [('files', (f'random_{i}.jpg', image_data, 'image/jpeg')) for i in range(batch_size)]
        response = requests.post(url, files=files)
        latency = time.time() - start_time
        return response.status_code, latency
    except Exception as e:
        print(f"Request failed: {e}")
        return 0, 0

def stress_test(url, num_requests, concurrency, batch_size):
    print(f"Starting stress test on {url}")
    print(f"Total requests: {num_requests}, Concurrency: {concurrency}, Batch Size: {batch_size}")
    
    # Pre-generate image to avoid measuring generation time
    image_buf = generate_random_image()
    image_data = image_buf.read()
    
    start_time = time.time()
    latencies = []
    success_count = 0
    error_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(send_request, url, image_data, batch_size): i for i in range(num_requests)}
        
        for future in concurrent.futures.as_completed(futures):
            status, latency = future.result()
            if status == 200:
                success_count += 1
                latencies.append(latency)
            else:
                error_count += 1

    total_time = time.time() - start_time
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = (success_count * batch_size) / total_time # Images per second
    req_throughput = success_count / total_time # Requests per second

    result = {
        "total_time": total_time,
        "successful_requests": success_count,
        "failed_requests": error_count,
        "request_throughput": req_throughput,
        "image_throughput": throughput,
        "average_latency": avg_latency
    }
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test the inference server.")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="Server URL")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per request")
    
    args = parser.parse_args()
    
    stress_test(args.url, args.requests, args.concurrency, args.batch_size)
