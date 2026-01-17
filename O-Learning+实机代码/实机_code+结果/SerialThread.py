import serial
from queue import Queue, Empty
from time import sleep
from threading import Thread

from ReadDataParser import ReadDataParser, Fh0cBase
from CommandConstructor import CommandConstructor
from QueueSignal import QueueSignal


class ThreadLocal:
    """used by thead"""
    latest_cmd: bytearray = None
    q: Queue = None
    s: serial.Serial = None
    t: Thread = None
    exit_queue: Queue = Queue()

    rdp: ReadDataParser = None

    def __init__(self):
        pass

    pass


def task_write(thead_local: ThreadLocal):
    """
    serial port write worker thread, this function do the write task in a independent thread.
    this function must run in a independent thread
    :param thead_local: the thread local data object
    """
    print("task_write")
    while True:
        sleep(0.1)
        try:
            # check if exit signal comes from main thread
            if thead_local.exit_queue.get(block=False) is QueueSignal.SHUTDOWN:
                break
        except Empty:
            pass
        try:
            # check if new command comes from CommandConstructor
            d = thead_local.q.get(block=False, timeout=-1)
            if isinstance(d, tuple):
                print("Tuple:", (d[0], d[1].hex(' ')))
                if d[0] is QueueSignal.CMD and len(d[1]) > 0:
                    thead_local.latest_cmd = d[1]
                    pass
                pass
        except Empty:
            pass
        # send cmd
        # must send multi time to ensure command are sent successfully
        if thead_local.latest_cmd is not None and len(thead_local.latest_cmd) > 0:
            # print("write:", thead_local.latest_cmd)
            thead_local.s.write(thead_local.latest_cmd)
        pass
    print("task_write done.")
    pass


def task_read(thead_local: ThreadLocal):
    """
    serial port read worker thread, this function do the write task in a independent thread.
    this function must run in a independent thread
    :param thead_local: the thread local data object
    """
    print("task_read\n")
    while True:
        sleep(0.1)
        try:
            # check if exit signal comes from main thread
            if thead_local.exit_queue.get(block=False) is QueueSignal.SHUTDOWN:
                break
        except Empty:
            pass
        # read from serial port
        d = thead_local.s.read(65535)
        # print("read:", d.hex())
        # write new data to ReadDataParser
        thead_local.rdp.push(d)
    print("task_read done.")
    pass


class SerialThreadCore:
    """
    the core function of serial control , this can run in main thread or a independent thread.
    """

    s: serial.Serial = None
    port: str = None
    thead_local_write: ThreadLocal = None
    thead_local_read: ThreadLocal = None

    def __init__(self, port: str):
        self.port = port
        self.q_write: Queue = Queue()
        self.q_read: Queue = Queue()
        self.s = serial.Serial(port, baudrate=500000, timeout=0.01)

        self.thead_local_write = ThreadLocal()
        self.thead_local_write.q = self.q_write
        self.thead_local_write.s = self.s
        self.thead_local_write.t = Thread(target=task_write, args=(self.thead_local_write,))

        self.thead_local_read = ThreadLocal()
        self.thead_local_read.q = self.q_read
        self.thead_local_read.s = self.s
        self.thead_local_read.rdp = ReadDataParser(self.thead_local_read.q)
        self.thead_local_read.t = Thread(target=task_read, args=(self.thead_local_read,))

        self.thead_local_write.t.start()
        self.thead_local_read.t.start()

    def shutdown(self):
        """
        this function safely shutdown serial port and the control thread,
        it does all the cleanup task.
        """
        # send exit signal comes to worker thread
        self.thead_local_write.exit_queue.put(QueueSignal.SHUTDOWN)
        self.thead_local_read.exit_queue.put(QueueSignal.SHUTDOWN)
        self.thead_local_write.t.join()
        self.thead_local_read.t.join()
        self.s.close()
        pass

    def fh0c_base(self) -> Fh0cBase:
        return self.thead_local_read.rdp.get_fh0c_base()
    pass


class SerialThread(SerialThreadCore):
    """
    this class extends SerialThreadCore, and implements more useful functions
    """

    ss: CommandConstructor = None

    def __init__(self, port: str):
        super().__init__(port)
        self.ss = CommandConstructor(self.thead_local_write.q)
        print("ss", self.ss)
        pass

    def send(self) -> CommandConstructor:
        return self.ss

    pass


# TODO manual test code on here
if __name__ == '__main__':
    st=SerialThread("COM6")
    st.send().takeoff(50)
    sleep(3)
    st.send().takeoff(50)
    sleep(2)
    st.send().led(0, 1, 1, 1)
    sleep(2)
    st.send().led(2, 1, 1, 1)
    sleep(3)
    st.send().speed(20)
    sleep(3)
    st.send().high(150)
    sleep(6)
    st.send().move(7, 80)
    sleep(6)
    st.send().rotate(90)
    sleep(6)
    st.send().back(60)
    sleep(6)
    st.send().arrive(60,60,60)
    sleep(6)
    st.send().high(120)
    sleep(6)
    st.send().flip_left(1)
    sleep(6)
    st.send().land()
    sleep(2)
    st.shutdown()
    sleep(2)

