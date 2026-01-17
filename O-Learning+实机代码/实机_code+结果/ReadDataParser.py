"""
这个文件负责解析从飞机串口发来的二进制数据
"""
from queue import Queue
from struct import pack, unpack, pack_into, unpack_from
from typing import List, Dict, Any, Tuple, Union, Literal
from dataclasses import dataclass
from threading import Lock

Header_Base_Info = b'\xAA\x0D\x07'
Header_Vision_Sensor_info = b'\xAA\x19\x30'
Header_Sensor_info = b'\xAA\x14\x01'
Header_Others = b'\xAA\x09\xf1'
Header_Others_Hardware_Info = b'\xAA\x00\x00'
Header_Others_MultiSetting_Info = b'\xAA\x00\x04'
Header_Others_SingleSetting_Info = b'\xAA\x00\x05'
Header_Fh0cBase = b'\xAA\x1b\x01'


#  typedef struct
#  {
#      u8 id;           //编号
#      u8 vol;          //电压(例如：37表示3.7V)
#      u8 ssi;          //信号强度
#      u16 state;       //传感器状态
#      u16 setting;     //设置状态
#
#      struct {
#          u8 flag;      //bit0=1，前方有线；bit1=1，后方有线；bit2=1，左方有线；bit3=1，右方有线；
#                        //bit4=1，检测到标签；bit5=1，检测到点；bit6=1，检测到二维码；bit7=1，检测到条形码。
#
#          u16 tagId;    //如果检测到标签，这里就是检测到的标签号。
#
#          s8 x0,y0;     //如果检测到标签，这里就是标签的坐标，
#                        //如果检测到线，这里就是横线的坐标，
#                        //如果检测到点，这里就是点的坐标。
#
#          s8 x1,y1;     //如果检测到线，这里就是竖线的坐标。
#      }mv;
#
#      //14
#      struct {
#          s8 qual;     // 光流数据可靠性指数
#          s8 x,y;      // 光流x/y轴数据
#      }flow;
#      //17
#      struct {
#          s16 x,y,z;   // 横滚/俯仰/航向角度
#      }imu;
#      //23
#      s16 high;
#  }
#  basicSensor_t;

@dataclass
class Fh0cBase:
    id: int  # u8
    vol: int  # u8
    ssi: int  # u8
    state: int  # u16
    setting: int  # u16

    mv_flag: int  # u8
    mv_tagId: int  # u16
    mv_x0: int  # s8
    mv_y0: int  # s8
    mv_x1: int  # s8
    mv_y1: int  # s8

    flow_qual: int  # s8
    flow_x: int  # s8
    flow_y: int  # s8

    imu: Tuple[int, int, int]  # s16 x,y,z
    high: int  # s16 高度


class ReadDataParser:
    """
    this class Parse data that comes from serial port.
    """

    read_buffer: bytearray = bytearray()
    q: Queue = None
    m_fh0c_base: Fh0cBase = None
    m_info_lock: Lock = Lock()

    def __init__(self, q_read):
        self.q = q_read
        pass

    def push(self, data: Union[bytearray, bytes]):
        """
        this method are used by `task_read` thread, it add new bytearray comes from serial port.
        """
        self.read_buffer = self.read_buffer + data
        self.try_parse()
        pass

    def try_parse(self):
        """
        this method do the core task that split bytearray buffer stream into packages.
        """
        # print(self.read_buffer)
        while len(self.read_buffer) > 3:
            header = self.read_buffer[0:3]
            size = header[1]
            if len(self.read_buffer) <= size + 3:
                break

            # print("header", header, size, header[0], header[1], header[2])
            if header == Header_Fh0cBase:
                data = self.read_buffer[0: size + 3]
                # print("fh0c_base", 0, size, len(data), data)
                self.fh0c_base(data)
                pass

            self.read_buffer = self.read_buffer[size + 3:]
            # if len(self.read_buffer) > size + 3:
            #     self.read_buffer = self.read_buffer[size + 3:]
            #     self.try_parse()
            #     pass
            pass
        pass

    def fh0c_base(self, data: bytearray):
        # print("fh0c_base", data.hex(' '))
        params = data[2:len(data) - 1]
        m_fh0c_base = Fh0cBase(
            id=unpack_from("!B", params, 1)[0],
            vol=unpack_from("!B", params, 2)[0],
            ssi=unpack_from("!B", params, 3)[0],
            state=unpack_from("!H", params, 4)[0],
            setting=unpack_from("!H", params, 6)[0],
            mv_flag=unpack_from("!B", params, 8)[0],
            mv_tagId=unpack_from("!H", params, 9)[0],
            mv_x0=unpack_from("!b", params, 11)[0],
            mv_y0=unpack_from("!b", params, 12)[0],
            mv_x1=unpack_from("!b", params, 13)[0],
            mv_y1=unpack_from("!b", params, 14)[0],
            flow_qual=unpack_from("!b", params, 15)[0],
            flow_x=unpack_from("!b", params, 16)[0],
            flow_y=unpack_from("!b", params, 17)[0],
            imu=(unpack_from("!h", params, 18)[0],
                 unpack_from("!h", params, 20)[0],
                 unpack_from("!h", params, 22)[0]),
            high=unpack_from("!h", params, 24)[0],
        )
        with self.m_info_lock:
            self.m_fh0c_base = m_fh0c_base
            pass
        # print("self._fh0c_base", m_fh0c_base)
        pass

    def get_fh0c_base(self):
        with self.m_info_lock:
            return self.m_fh0c_base
    pass
