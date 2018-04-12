from __future__ import print_function
from ctypes import c_bool
import multiprocessing as mp

try:
    from queue import Full as QFullExcept
    from queue import Empty as QEmptyExcept
except ImportError:
    from Queue import Full as QFullExcept
    from Queue import Empty as QEmptyExcept

import numpy as np

class MPData(object):
    """
        Handles multi-process data generation.
        
        Operation:
        - call start() to start the data generation.
        - call get() (blocking) to read one sample
        - call reset() to stop data generation
    """
    def __init__(self, num_processes, max_queue_size, fn):
        self.queue = mp.Queue(maxsize=int(max_queue_size))
        self.alive = mp.Value(c_bool, False, lock=False)
        self.num_proc = num_processes
        self.proc = list()
        self.fn = fn
    
    def start(self):
        self._init_proc()
        
    @staticmethod
    def _proc_loop(proc_id, alive, queue, fn):
        """
            proc_id: int
                Process id
            alive: multiprocessing.Value
                variable for signaling whether process should continue or not
            queue: multiprocessing.Queue
                queue for passing data back
            fn: function
                func obj that returns a sample to be pushed into the queue
        """
        print('proc {} started'.format(proc_id))
        try:
            while alive.value:
                data = fn()
                put_success = False
                while alive.value and not put_success:
                    try:
                        queue.put(data, timeout=0.5)
                        put_success = True
                    except QFullExcept:
                        pass
        except KeyboardInterrupt:
            print("W: interrupt received, stopping process {} ...".format(proc_id))
        print("closing process {}".format(proc_id))
        queue.close()
        
    def _init_proc(self):
        if not self.proc:
            self.proc = [mp.Process(target=self._proc_loop, args=(i, self.alive, self.queue, self.fn)) for i in range(self.num_proc)]
            self.alive.value = True
            for p in self.proc:
                p.start()
    
    def get(self):
        self._init_proc()
        return self.queue.get()
    
    def reset(self):
        self.alive.value = False
        qsize = 0
        try:
            while True:
                self.queue.get(timeout=0.1)
                qsize += 1
        except QEmptyExcept:
            pass
        print("Queue size on reset: {}".format(qsize))
        for i, p in enumerate(self.proc):
            p.join()
        self.proc.clear()
    

