
Default `concurrent.futures.ProcessPoolExecutor` makes it hard to maintain stateful workers (e.g., workers each with a model loaded in GPU memory).
This library provides a simple interface to create a pool of stateful workers that can execute tasks in parallel across multiple processes. 

```text
+-----------------------+              +----------------------+
|     Main Process      |              |     Process Pool     |
| +--------+ +--------+ |              | +------------------+ |
| | Thread | | Thread | | --- Task --> | | Worker (Process) | |
| |   1    | |   2    | | <-- Resp --- | +------------------+ |
| +--------+ +--------+ |              | +------------------+ |
|      ...     ...      |              | | Worker (Process) | |
| +--------+ +--------+ |              | +------------------+ |
| | Thread | | Thread | | --- Task --> |         ...          |
| |  N-1   | |   N    | | <-- Resp --- | +------------------+ |
| +--------+ +--------+ |              | | Worker (Process) | |
|                       |              | +------------------+ |
+-----------------------+              +----------------------+
```

Installation:
```bash
pip install stateful-pool
```

Following is a simple example of how to use the library to create a pool of workers each maintaining its own state (in this case, a GPU ID). 
A task is submitted to the pool, and the result is retrieved in a blocking manner. 
In practice, you would likely want to submit tasks from a separate thread to avoid blocking.

```python
from stateful_pool import SPool, SWorker
import time, random

class SquareWorker(SWorker):
    gpu_id: int

    def spawn(self, gpu_id):
        self.gpu_id = gpu_id
        return f"Worker initialized on GPU {gpu_id}"
    
    def execute(self, value):
        time.sleep(random.uniform(0.1, 1.0))
        return f"[Execute] Square of {value} is {value * value} (computed on GPU {self.gpu_id})"

if __name__ == "__main__":

    with SPool(SquareWorker, queue_size=100) as pool:
        s1 = pool.spawn(gpu_id=1)
        s2 = pool.spawn(gpu_id=2)
        print(f"{s1}, {s2}")

        res = pool.execute(100)
        print(res)
```