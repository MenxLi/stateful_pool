import asyncio
import io
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

# Global State
workers = []

class ModelWorker:
    def __init__(self, device_str):
        self.device = torch.device(device_str)
        self.model, self.preprocess = model_init.initialize_model(device_str)
        print(f"Ready on {device_str}")

    def process(self, images_bytes_list):
        try:
            results = []
            batch_tensors = []
            
            for image_bytes in images_bytes_list:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tensor = self.preprocess(image)
                batch_tensors.append(tensor)
            
            if batch_tensors:
                batch = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    predictions = self.model(batch).softmax(1)
                    
                    for i in range(len(batch_tensors)):
                        prediction = predictions[i]
                        class_id = prediction.argmax().item()
                        score = prediction[class_id].item()
                        
                        results.append({
                            "class_id": class_id,
                            "confidence": score,
                        })
            return results
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workers
    
    assert torch.cuda.is_available(), "No GPUs found. Please run on a machine with CUDA-enabled GPUs."
    num_gpus = torch.cuda.device_count()
    print(f"Server: Found {num_gpus} GPUs.")
    devices = [f"cuda:{i}" for i in range(num_gpus)]
        
    for dev in devices:
        workers.append(ModelWorker(dev))

    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    contents_list = []
    for file in files:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File provided is not an image.")
        contents = await file.read()
        contents_list.append(contents)
    
    if not workers:
         raise HTTPException(status_code=503, detail="No workers available")
             
    # Random load balancing
    worker = random.choice(workers)
    
    # Run inference in a thread to keep the event loop non-blocking
    result = await asyncio.to_thread(worker.process, contents_list)
            
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
