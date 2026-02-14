
Default `ProcessPoolExecutor` makes it hard to maintain stateful workers (e.g., workers each with a model loaded in GPU memory).
This library lets you create a pool of stateful workers to run tasks in parallel across processes.

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

Following is an example of how to spawn workers (each assigned to several GPUs), and execute tasks on them.
The result is retrieved in a blocking manner. 

```python
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
```

In practice, you would likely want to submit tasks from a separate thread to avoid blocking (refer to `example.py`). 
The implementation is thread-safe.