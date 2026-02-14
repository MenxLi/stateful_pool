from concurrent.futures import ThreadPoolExecutor
from stateful_pool import SPool, SWorker
import time, random

# will run in another process
class SquareWorker(SWorker):
    gpu_ids: list[int]

    def spawn(self, gpu_ids: list[int]):
        self.gpu_ids = gpu_ids
        return f"Worker initialized on GPU: {self.gpu_ids}"
    
    def execute(self, value):
        time.sleep(random.uniform(0.1, 1.0))
        return f"[Execute] Square of {value} is {value * value} (computed on GPU: {self.gpu_ids})"

if __name__ == "__main__":

    with SPool(SquareWorker, queue_size=100) as pool:
        # spawn a worker, return value can be captured
        s = pool.spawn(gpu_ids=[0, 1])
        print(f"{s}")

        # submit a single task and wait for result
        r = pool.execute(100)
        print(r)
    
    # manually manage the pool without context manager
    print("=" * 20, "Manual Pool Management", "=" * 20)
    pool = SPool(SquareWorker)
    pool.spawn(gpu_ids=[0, 1])
    res = pool.execute(200)
    print(res)
    pool.shutdown()

    # submit multiple tasks concurrently, thread pool will handle scheduling and execution
    print("=" * 20, "Concurrent Task Submission", "=" * 20)
    with SPool(SquareWorker) as pool:
        spawn_futures = [pool.submit_spawn(gpu_ids=[i, i+1]) for i in range(0, 4, 2)]
        for f in spawn_futures:
            print(f.result())
        
        execute_futures = [pool.submit_execute(i) for i in range(4)]
        for f in execute_futures:
            print(f.result())
    