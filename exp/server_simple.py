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
from torchvision.models import resnet50, ResNet50_Weights

# Global State
workers = []

class ModelWorker:
    def __init__(self, device_str):
        print(f"Initializing model on {device_str}...")
        self.device = torch.device(device_str)
        self.weights = ResNet50_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.model = resnet50(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()
        self.categories = self.weights.meta["categories"]
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
                        category_name = self.categories[class_id]
                        
                        results.append({
                            "class_id": class_id,
                            "class_name": category_name,
                            "confidence": score,
                            "device": str(self.device),
                            "worker_pid": os.getpid()
                        })
            return results
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workers
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Server: Found {num_gpus} GPUs.")
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    else:
        print("Server: No GPUs found. Falling back to CPU.")
        devices = ["cpu"]
        
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
