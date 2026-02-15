import asyncio
import io
import uvicorn
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from PIL import Image
from stateful_pool import SPool, SWorker

try:
    import model_init
except ImportError:
    from exp import model_init

class ModelWorker(SWorker):
    def spawn(self, device_str: str):
        self.device = torch.device(device_str)
        self.model, self.preprocess = model_init.initialize_model(device_str)
        return f"Worker initialized on {device_str}"

    def execute(self, images_bytes_list: List[bytes]):
        batch_tensors = []
        for image_bytes in images_bytes_list:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.preprocess(image)
            batch_tensors.append(tensor)
        
        if not batch_tensors:
            return []

        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(batch).softmax(1)
            
            results = []
            for i in range(len(batch_tensors)):
                prediction = predictions[i]
                class_id = prediction.argmax().item()
                score = prediction[class_id].item()
                
                results.append({
                    "class_id": class_id,
                    "confidence": score,
                })
        return results

pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    if not torch.cuda.is_available():
        print("[Error] No GPUs found. ")
        exit(1)

    pool = SPool(ModelWorker, queue_size=100)
    futures = []
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Spawning workers...")
    for i in range(num_gpus):
        futures.append(pool.submit_spawn(device_str=f"cuda:{i}"))

    for f in futures:
        try:
            print(f.result())
        except Exception as e:
            print(f"Worker spawn failed: {e}")
            
    yield
    
    if pool: 
        pool.shutdown()

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

    try:
        assert pool
        future = pool.submit_execute(contents_list)
        result = await asyncio.wrap_future(future)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import multiprocessing
    # Set the start method for multiprocessing to 'spawn' to be compatible with CUDA
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    uvicorn.run(app, host="0.0.0.0", port=8000)