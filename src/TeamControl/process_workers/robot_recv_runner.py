from multiprocessing import Queue
from TeamControl.process_workers.worker import BaseWorker
from TeamControl.network.receiver import Receiver


RECV_PORT = 50513


class RobotRecv(BaseWorker):
    def __init__(self, is_running, logger):
        super().__init__(is_running, logger)
        self.recv = Receiver(is_running=is_running, port=RECV_PORT)
        self._queue: Queue | None = None

    def setup(self, recv_queue: Queue):
        self._queue = recv_queue
        super().setup()

    def step(self):
        data, addr = self.recv.listen()
        if data is not None and self._queue is not None:
            self._queue.put((data, addr))
