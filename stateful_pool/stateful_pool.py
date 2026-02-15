import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional
import multiprocessing as mp
from threading import Event, Thread, Lock
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T')
SR = TypeVar('SR')
ER = TypeVar('ER')

class SWorker(ABC, Generic[SR, ER]):
    """
    The entire class will be run in a separate process, 
    so it should not have any shared state with the main process.
    """

    @abstractmethod
    def spawn(self, *args, **kwargs) -> SR:
        """ 
        Setup the worker on process start. 
        This method is called once when the worker process is initialized.
        The parameters will be passed from the SPool.spawn method.
        This method should return a value that will be passed back to the main process,
        or raise an exception if setup fails.
        """
        ...
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> ER:
        """ 
        Run a task on the worker. 
        This method is called for each task that needs to be processed by the worker.
        """
        ...

@dataclass
class _PendingResult(Generic[ER]):
    event: Event
    result: Optional[ER | Exception] = None

@dataclass
class _ExecutorState(Generic[ER]):
    ps: list[mp.Process]
    task_id_counter: int
    pending_results: dict[int, _PendingResult[ER]]

class _Guarded(Generic[T]):
    def __init__(self, value: T):
        self._value = value
        self._lock = Lock()
    
    def __enter__(self):
        self._lock.acquire()
        return self._value
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

class SPool(Generic[SR, ER]):
    submit_queue: "mp.Queue[tuple[int, Optional[tuple[tuple, dict]]]]"
    result_queue: "mp.Queue[tuple[int, Optional[ER | Exception]]]"
    state: _Guarded[_ExecutorState[ER]]

    def __init__(
        self, 
        worker_cls: type[SWorker[SR, ER]], 
        queue_size = None, 
        submission_timeout: Optional[float] = None,
        execution_timeout: Optional[float] = None, 
        thread_pool: Optional[ThreadPoolExecutor] = None
        ):
        self.worker_cls = worker_cls
        self.submit_queue = mp.Queue(0 if queue_size is None else queue_size)
        self.result_queue = mp.Queue()
        self.state = _Guarded(_ExecutorState(
            ps=[],
            task_id_counter=0, 
            pending_results={},
        ))

        self.listen_thread = self._listen()
        self.submission_timeout = submission_timeout
        self.execution_timeout = execution_timeout

        self._thread_pool = thread_pool or ThreadPoolExecutor()
        self._external_thread_pool = thread_pool is not None
    
    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor()
        return self._thread_pool
    
    @staticmethod
    def _worker_process(
        startup_queue: mp.Queue,
        worker_cls: type[SWorker[SR, ER]],
        submit_queue: mp.Queue,
        result_queue: mp.Queue,
        *args, **kwargs
        ):
        worker = worker_cls()
        try:
            r = worker.spawn(*args, **kwargs)
            startup_queue.put(r)    # Signal that setup is complete
        except Exception as e:
            startup_queue.put(e)
            return

        while True:
            task_id, task = submit_queue.get()
            if task is None:  # Sentinel value to signal shutdown
                break
            try:
                result = worker.execute(*task[0], **task[1])
            except Exception as e:
                result = e  # Capture exceptions to return them as results
            result_queue.put((task_id, result))
    
    def _listen(self) -> Thread:
        def _listen_loop():
            while True:
                try:
                    task_id, result = self.result_queue.get()
                    if result is None:
                        break
                    with self.state as state:
                        if task_id in state.pending_results:
                            # task_id may not be present if the task timed out and was removed
                            state.pending_results[task_id].result = result
                            state.pending_results[task_id].event.set()
                except Exception as e:
                    raise e
        t = Thread(target=_listen_loop, daemon=True)
        t.start()
        return t
    
    def spawn(self, *args, **kwargs) -> SR:
        """ 
        Add a worker to the pool. 
        Inputs are passed to the worker's setup method.
        Will wait for the worker to signal that setup is complete before returning, 
        and will raise any exceptions that occur during setup.

        This method creates a new process for the worker, 
        it must be guarded by `if __name__ == "__main__":` to avoid infinite process spawning.
        """
        startup_queue = mp.Queue()
        self_args = (startup_queue, self.worker_cls, self.submit_queue, self.result_queue)
        p = mp.Process(target=self._worker_process, args=self_args + args, kwargs=kwargs, daemon=True)
        p.start()
        
        # Wait for the worker to signal that setup is complete or failed
        setup_result = startup_queue.get()
        startup_queue.close()
        startup_queue.join_thread()
        if isinstance(setup_result, Exception):
            p.terminate()
            p.join()
            raise setup_result

        with self.state as state:
            state.ps.append(p)
        return setup_result

    def execute(self, *args, **kwargs) -> ER:
        """ 
        Submit a task to the pool for processing.  
        This should typically be called from thread, will block until the task is completed.
        """
        with self.state as state:
            if not state.ps:
                raise RuntimeError("Cannot execute task: no workers in the pool.")

            state.task_id_counter += 1
            task_id = state.task_id_counter
            state.pending_results[task_id] = _PendingResult(event=Event())

        try:
            self.submit_queue.put(
                (task_id, (args, kwargs)), 
                timeout=self.submission_timeout
                )
            with self.state as state:
                pending_result = state.pending_results[task_id]
            
            completed = pending_result.event.wait(timeout=self.execution_timeout)
            if not completed:
                raise TimeoutError(f"Task {task_id} timed out after {self.execution_timeout} seconds.")

            res = pending_result.result
        except Exception as e:
            raise e
        finally:
            with self.state as state:
                state.pending_results.pop(task_id, None)

        if isinstance(res, Exception):
            raise res
        return res # type: ignore
    
    def shutdown(self) -> None:
        """ Shutdown the pool and all worker processes. """
        import queue
        if not self.listen_thread.is_alive():
            return  # Already shutdown

        with self.state as state:
            for _ in state.ps:
                try:
                    self.submit_queue.put((-1, None), timeout=0.1)
                except queue.Full:
                    # will be handled by following join/terminate logic
                    pass
                    
            self.result_queue.put((-1, None))

            for p in state.ps:
                p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()
                    p.join()

        if self.listen_thread.is_alive():
            self.listen_thread.join()
        self.submit_queue.close()
        self.result_queue.close()
        self.submit_queue.join_thread()
        self.result_queue.join_thread()
        
        with self.state as state:
            state.ps.clear()
            state.pending_results.clear()
        
        if not self._external_thread_pool and self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
    
    def submit_spawn(self, *args, **kwargs):
        return self.thread_pool.submit(self.spawn, *args, **kwargs)
    
    def submit_execute(self, *args, **kwargs):
        return self.thread_pool.submit(self.execute, *args, **kwargs)
    
    async def async_spawn(self, *args, **kwargs):
        return await asyncio.wrap_future(self.submit_spawn(*args, **kwargs))
    
    async def async_execute(self, *args, **kwargs):
        return await asyncio.wrap_future(self.submit_execute(*args, **kwargs))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()