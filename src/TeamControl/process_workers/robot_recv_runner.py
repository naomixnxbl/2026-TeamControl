from multiprocessing import Queue
from TeamControl.process_workers.worker import BaseWorker
from TeamControl.network.receiver import Receiver


RECV_PORT = 50513


class RobotRecv(BaseWorker):
    def __init__(self, is_running, logger):
        super().__init__(is_running, logger)
        # Bind to 0.0.0.0 so packets arriving on ANY interface (robot LAN,
        # Wi-Fi, loopback) are accepted. Default _obtain_sys_ip() picks the
        # interface routing to 8.8.8.8, which on Windows is usually the
        # internet-facing one — robot packets would never arrive there.
        self.recv = Receiver(is_running=is_running, ip="0.0.0.0",
                             port=RECV_PORT)
        self._queue: Queue | None = None
        self.logger.info(
            f"[RobotRecv] listening on {self.recv.addr} "
            f"(ready={self.recv.is_ready})")

    def setup(self, recv_queue: Queue):
        self._queue = recv_queue
        super().setup()

    def step(self):
        data, addr = self.recv.listen()
        if data is not None and self._queue is not None:
            self._queue.put((data, addr))
