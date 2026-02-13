from concurrent.futures import ThreadPoolExecutor
from stateful_pool import SPool, SWorker
import time, random

# type annotations for [ST, T, RT] are:
# [spawn return type, execute argument type, execute return type]
class SquareWorker(SWorker[str, int, str]):
    gpu_id: int

    def spawn(self, gpu_id):
        self.gpu_id = gpu_id
        return f"Worker initialized on GPU {gpu_id}"
    
    def execute(self, task):
        time.sleep(random.uniform(0.1, 1.0))
        return f"[Execute] Square of {task} is {task * task} (computed on GPU {self.gpu_id})"

if __name__ == "__main__":

    with SPool(SquareWorker, queue_size=100) as pool:
        s1 = pool.spawn(gpu_id=1)
        s2 = pool.spawn(gpu_id=2)
        print(f"{s1}, {s2}")

        res = pool.execute(100)
        print(res)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(pool.execute, i) for i in range(10)]
            for i, future in enumerate(futures):
                res = future.result()
                print(res)
    