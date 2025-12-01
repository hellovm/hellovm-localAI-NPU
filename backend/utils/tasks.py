import threading
import uuid
import queue
import copy

class TaskStore:
    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()
        self._listeners = {}  # task_id -> list of queues

    def create(self, kind):
        task_id = str(uuid.uuid4())
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "kind": kind,
                "status": "pending",
                "progress": 0,
                "message": "",
                "result": None,
                "error": None,
            }
        return task_id

    def _notify(self, task_id, task_data):
        if task_id in self._listeners:
            # Create a copy to avoid race conditions if the listener modifies it (though queues are thread-safe)
            # We send a copy of the data
            data = copy.deepcopy(task_data)
            for q in self._listeners[task_id]:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass

    def subscribe(self, task_id):
        q = queue.Queue(maxsize=100)
        with self._lock:
            if task_id not in self._listeners:
                self._listeners[task_id] = []
            self._listeners[task_id].append(q)
            # Send current state immediately
            if task_id in self._tasks:
                q.put(copy.deepcopy(self._tasks[task_id]))
        return q

    def unsubscribe(self, task_id, q):
        with self._lock:
            if task_id in self._listeners:
                if q in self._listeners[task_id]:
                    self._listeners[task_id].remove(q)
                if not self._listeners[task_id]:
                    del self._listeners[task_id]

    def update(self, task_id, progress=None, status=None, message=None, result=None, error=None):
        with self._lock:
            t = self._tasks.get(task_id)
            if not t:
                return
            if progress is not None:
                t["progress"] = progress
            if status is not None:
                t["status"] = status
            if message is not None:
                t["message"] = message
            if result is not None:
                t["result"] = result
            if error is not None:
                t["error"] = error
            
            self._notify(task_id, t)

    def get(self, task_id):
        with self._lock:
            return self._tasks.get(task_id)

    def complete(self, task_id, result=None):
        with self._lock:
            t = self._tasks.get(task_id)
            if not t:
                return
            t["status"] = "completed"
            t["progress"] = 100
            t["result"] = result
            self._notify(task_id, t)

task_store = TaskStore()