import asyncio
import io
import uvicorn
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from stateful_pool import SPool, SWorker

class ResNetWorker(SWorker):
    def spawn(self, device_str: str):
        print(f"Initializing worker on {device_str}")
        self.device = torch.device(device_str)
        
        # Load model
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = self.weights.transforms()
        return f"Worker initialized on {device_str}"

    def execute(self, images_bytes_list: List[bytes]):
        try:
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
                    category_name = self.weights.meta["categories"][class_id]
                    
                    results.append({
                        "class_id": class_id,
                        "class_name": category_name,
                        "confidence": score,
                        "device": str(self.device)
                    })
                
            return results
        except Exception as e:
            return {"error": str(e)}

pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = SPool(ResNetWorker, queue_size=100)
    
    futures = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs. Spawning workers...")
        for i in range(num_gpus):
            futures.append(pool.submit_spawn(device_str=f"cuda:{i}"))
    else:
        print("No GPUs found. Spawning CPU workers...")
        futures.append(pool.submit_spawn(device_str="cpu"))

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
        
        if isinstance(result, dict) and "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
            
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