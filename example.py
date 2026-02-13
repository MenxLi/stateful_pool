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
        # return value from spawn can be captured
        s1 = pool.spawn(gpu_ids=[0, 1])
        s2 = pool.spawn(gpu_ids=[2, 3])
        print(f"{s1}, {s2}")

        # submit a single task and wait for result
        res = pool.execute(100)
        print(res)

        # submit multiple tasks concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(pool.execute, i) for i in range(10)]
            for i, future in enumerate(futures):
                res = future.result()
                print(res)
    