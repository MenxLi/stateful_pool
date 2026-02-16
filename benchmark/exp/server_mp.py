import asyncio
import io
import multiprocessing
import os
import random
import traceback
from typing import List
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from . import model_init

# Global Configuration
MAX_CONCURRENT_REQUESTS = torch.cuda.device_count() * 2 if torch.cuda.is_available() else 4

# Global State
processes: List[multiprocessing.Process] = []
parent_conns = []
locks: List[asyncio.Lock] = []
global_sem: asyncio.Semaphore = None #type: ignore

def model_worker(device_str: str, conn):
    """
    Worker process that initializes the model on a specific GPU/device
    and processes requests sent via the pipe.
    """
    try:
        # Initialize model
        model, preprocess = model_init.initialize_model(device_str)
        device = torch.device(device_str)
        
        while True:
            try:
                data = conn.recv()
            except EOFError:
                break
                
            if data is None:
                break
                
            try:
                images_bytes_list = data
                results = []
                
                batch_tensors = []
                for image_bytes in images_bytes_list:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    tensor = preprocess(image)
                    batch_tensors.append(tensor)
                
                batch = torch.stack(batch_tensors).to(device)
                
                with torch.no_grad():
                    predictions = model(batch).softmax(1)
                    
                    for i in range(len(batch_tensors)):
                        prediction = predictions[i]
                        class_id = prediction.argmax().item()
                        score = prediction[class_id].item()
                        
                        results.append({
                            "class_id": class_id,
                            "confidence": score,
                        })
                conn.send(results)
                
            except Exception as e:
                conn.send({"error": str(e)})

    except Exception as e:
        print(f"[Worker {os.getpid()}] Fatal initialization error: {e}")
        traceback.print_exc()
    finally:
        print(f"[Worker {os.getpid()}] Exiting...")


def send_to_worker(conn, data):
    try:
        conn.send(data)
        return conn.recv()
    except (EOFError, BrokenPipeError):
        return {"error": "Worker process unreachable (terminated?)"}
    except Exception as e:
        return {"error": f"Communication error: {e}"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_sem, processes, parent_conns, locks
    
    # Initialize global semaphore for limiting overall concurrency
    global_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Detect devices
    assert torch.cuda.is_available(), "No GPUs found. Please run on a machine with CUDA-enabled GPUs."
    num_gpus = torch.cuda.device_count()
    print(f"Master: Found {num_gpus} GPUs. Spawning workers...")
    devices = [f"cuda:{i}" for i in range(num_gpus)]
        
    for dev in devices:
        parent_conn, child_conn = multiprocessing.Pipe()
        
        p = multiprocessing.Process(target=model_worker, args=(dev, child_conn))
        p.start()
        
        processes.append(p)
        parent_conns.append(parent_conn)
        locks.append(asyncio.Lock())

    yield

    print("Master: Shutting down...")
    for conn in parent_conns:
        try:
            conn.send(None) # Sentinel
        except:
            pass
            
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    contents_list = []
    for file in files:
        assert file.content_type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File provided is not an image.")
        contents = await file.read()
        contents_list.append(contents)
    
    async with global_sem:
        # Random Dispatch strategy
        if not parent_conns:
             raise HTTPException(status_code=503, detail="No workers available")
             
        worker_idx = random.randrange(len(parent_conns))
        
        async with locks[worker_idx]:
            conn = parent_conns[worker_idx]
            result = await asyncio.to_thread(send_to_worker, conn, contents_list)
            
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)