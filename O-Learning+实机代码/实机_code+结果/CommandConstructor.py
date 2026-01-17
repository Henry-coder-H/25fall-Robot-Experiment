from queue import Queue
from struct import pack, unpack, pack_into, unpack_from
from enum import Enum

from QueueSignal import QueueSignal


class CmdType(Enum):
    MULTI_SETTINGS = 1
    SINGLE_SETTINGS = 2
    SINGLE_CONTROL = 3
    READ_SETTINGS = 4
    pass


class CommandConstructorCore:
    """
    this class impl core read functions
    """

    """
    last order was generate
    """
    order_last = 0
    """
    
    """
    q_write: Queue = None

    def _order_count(self):
        """
        the order_count generator, it generates new order number
        """
        self.order_last = (self.order_last + 1) % 127
        return self.order_last
        pass

    def _check_sum(self, header: bytearray, params: bytearray):
        """
        impl checksum algorithm
        """
        return bytearray([sum(header + params) & 0xFF])
        pass

    def sendCommand(self, data: bytearray):
        """
        direct do the command bytearray send task
        """
        self.q_write.put((QueueSignal.CMD, data), block=True)
        pass

    def __init__(self, q_write: Queue):
        self.q_write = q_write
        pass

    def join_cmd(self, type: CmdType, params: bytearray):
        """
        The join_cmd function concatenates the header, params and checksum to a cmd bytearray to let it become a valid serial cmd package.
        """
        header = bytearray(b'\xBB\x1D')  # HEAD=0xBB, LEN=0x1D(29)

        data_body = bytearray(29)
        data_body[0] = 0xF3  # FUN byte

        if len(params) > 13:
            raise ValueError(f"params length {len(params)} exceeds maximum 13 bytes")

        data_body[1:1+len(params)] = params

        # Copy params to cmd2 position
        data_body[14:14+len(params)] = params

        data_body[27] = 0x64  # volume = 100

        data_body[28] = 0x00  # reserved = 0

        checksum = self._check_sum(header, data_body)

        return header + data_body + checksum

    pass


class CommandConstructor(CommandConstructorCore):
    """
    this class extends CommandConstructorCore,
    it impl all functions that direct construct command .
    TODO Implement All Command Methods On This Class
    """

    def __init__(self, q_write: Queue):
        super().__init__(q_write)
        pass

    def led(self, mode, r, g, b):
        if mode < 0 or mode > 2:
            raise ValueError("mode illegal", mode)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 13)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<h', params, 3, mode)
        pack_into('<B', params, 5, r)
        pack_into('<B', params, 6, g)
        pack_into('<B', params, 7, b)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("led", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def takeoff(self, high: int, ):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 0)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<h', params, 3, high)
        pack_into('<B', params, 5, 50)
        pack_into('<B', params, 6, 0)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("take_off", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def land(self, ):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 254)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 0)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("land", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def move(self, direction: int, distance: int): #控制无人机移动
        if direction < 0 or direction > 10:
            raise ValueError("direction illegal", direction)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 5)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, direction+1)
        pack_into('<h', params, 4, distance)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("move", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def up(self, distance: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 5)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 5)
        pack_into('<h', params, 4, distance)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("up", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def down(self, distance: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 5)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 6)
        pack_into('<h', params, 4, distance)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("back", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def forward(self, distance: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 5)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 1)
        pack_into('<h', params, 4, distance)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("forward", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def back(self, distance: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 5)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 2)
        pack_into('<h', params, 4, distance)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("back", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def left(self, distance: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 5)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 3)
        pack_into('<h', params, 4, distance)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("left", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def right(self, distance: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 5)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 4)
        pack_into('<h', params, 4, distance)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("right", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def flip(self, direction: int, circle: int): #控制无人机翻滚
        if direction < 1 or direction > 4:
            raise ValueError("direction illegal", direction)
        if circle != 1 and circle != 2:
            raise ValueError("circle illegal", circle)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 12)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, direction)
        pack_into('<B', params, 4, circle)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("flip", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def flip_forward(self, circle: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 12)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 1)
        pack_into('<B', params, 4, circle)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("flip_forward", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def flip_back(self, circle: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 12)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 2)
        pack_into('<B', params, 4, circle)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("flip_back", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def flip_left(self, circle: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 12)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 3)
        pack_into('<B', params, 4, circle)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("flip_left", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def flip_right(self, circle: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 12)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 4)
        pack_into('<B', params, 4, circle)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("flip_right", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def arrive(self, x: int, y: int, z: int): #直达
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 9)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<h', params, 3, x)
        pack_into('<h', params, 5, y)
        pack_into('<h', params, 7, z)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("arrive", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def rotate(self, degree: int): #控制无人机自旋
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 10)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<h', params, 3, degree)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("rotate", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def speed(self, speed: int): #设置无人机速度
        if speed < 0 or speed > 200:
            raise ValueError("speed illegal", speed)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 2)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<h', params, 3, speed)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("speed", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def high(self, high: int):
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 11)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<h', params, 3, high)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("high", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def airplane_mode(self, mode: int): #切换无人机飞行模式
        if mode < 1 or mode > 4:
            raise ValueError("mode illegal", mode)
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 1)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, mode)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("airplane_mode", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def hovering(self, ): #悬停
        params = bytearray(13)
        pack_into('<B', params, 0, 0x00)
        pack_into('<B', params, 1, 254)
        pack_into('<B', params, 2, self._order_count())
        pack_into('<B', params, 3, 4)
        cmd = self.join_cmd(CmdType.SINGLE_CONTROL, params)
        print("hovering", cmd.hex(' '))
        self.sendCommand(cmd)
        pass

    def read_multi_setting(self):
        return self.read_setting(0x02)

    def read_single_setting(self):
        return self.read_setting(0x04)

    def read_hardware_setting(self):
        return self.read_setting(0xA0)

    def read_setting(self, mode: int):
        params = bytearray(1)
        pack_into("!B", params, 0, mode)
        cmd = self.join_cmd(CmdType.READ_SETTINGS, params)
        print("cmd", cmd.hex(' '))
        self.sendCommand(cmd)
        pass
    pass
