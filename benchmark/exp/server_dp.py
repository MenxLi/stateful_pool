import asyncio
import io
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from typing import List, Callable
from contextlib import asynccontextmanager
from asyncio import Semaphore
from . import model_init

# Global State
model: torch.nn.Module
preprocess: Callable
device = None
sem = Semaphore(1) # limit concurrent access to the model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocess, device
    
    assert torch.cuda.is_available(), "No GPUs found."
    num_gpus = torch.cuda.device_count()
    print(f"Server: Found {num_gpus} GPUs. Using DataParallel.")

    base_model, preprocess_fn = model_init.initialize_model("cuda:0")
    
    if num_gpus > 1:
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model
        
    preprocess = preprocess_fn
    device = torch.device("cuda:0") # DataParallel inputs usually go to the first device
    print("Model initialized and wrapped in DataParallel.")
    yield

app = FastAPI(lifespan=lifespan)

def process_batch(images_bytes_list: List[bytes]):
    batch_tensors = []
    for img_bytes in images_bytes_list:
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            tensor = preprocess(image)
            batch_tensors.append(tensor)
        except Exception as e:
            print(f"Error processing image: {e}")
            raise e

    if not batch_tensors:
        return []

    batch = torch.stack(batch_tensors).to(device, non_blocking=True)
    
    with torch.no_grad():
        predictions = model(batch).softmax(1)

    results = []
    predictions = predictions.cpu()
    
    for i in range(len(batch_tensors)):
        prediction = predictions[i]
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        results.append({
            "class_id": class_id,
            "confidence": score,
        })
    return results

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    contents_list = []
    for file in files:
        if file.content_type and not file.content_type.startswith("image/"):
             # Just warning/skipping or error? simple server raises error.
            pass 
        contents = await file.read()
        contents_list.append(contents)
    
    if not contents_list:
        raise HTTPException(status_code=400, detail="No images provided")

    # Run inference in a thread to verify async/await behavior doesn't block
    async with sem:
        return await asyncio.to_thread(process_batch, contents_list)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
