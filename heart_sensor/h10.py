import asyncio
import struct
import numpy as np
import datetime

from bleak import BleakClient
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

"""
Polar Ref: https://github.com/polarofficial/polar-ble-sdk/blob/master/technical_documentation/Polar_Measurement_Data_Specification.pdf
HR measurement Ref: https://github.com/oesmith/gatt-xml/blob/master/org.bluetooth.characteristic.heart_rate_measurement.xml 
"""

class H10:
    FUNCTIONS = {
        'ECG': 0, 
        'PPG': 1, 
        'Acceleration':2, 
        'PP interval':4,
        'Gryoscope':5,
        'Magnetometer': 6,
    }

    """ Predefined UUID (Universal Unique Identifier) mapping are based on Heart Rate GATT service Protocol that most
    Fitness/Heart Rate device manufacturer follow (Polar H10 in this case) to obtain a specific response input from 
    the device acting as an API """
    DEVICE_NAME_UUID = "00002a00-0000-1000-8000-00805f9b34fb"
    MODEL_NBR_UUID = "00002a24-0000-1000-8000-00805f9b34fb"
    MANUFACTURER_NAME_UUID = "00002a29-0000-1000-8000-00805f9b34fb"
    BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
    HR_MEASUREMENT = "00002A37-0000-1000-8000-00805f9b34fb"
    PMD_CONTROL = "FB005C81-02E7-F387-1CAD-8ACD2D8DF0C8"
    PMD_DATA = "FB005C82-02E7-F387-1CAD-8ACD2D8DF0C8"

    """
    UUID for Request of ECG Stream
    02 = start measurement, 00 = measurement type ecg
    00 01 82 00 = set sampling rate to 0x82 (130 Hz)
    01 01 0E 00 = set resolution to 0x0E (14 bit)
    """
    ECG_WRITE = bytearray([0x02, 0x00, 0x00, 0x01, 0x82, 0x00, 0x01, 0x01, 0x0E, 0x00])

    def __init__(self, addr, task=None, data_only=False):
        self.addr = addr
        self.fig = None
        self.sys_start_time = None
        self.sensor_start_time = None
        self.data = None
        self.data_only = data_only
        if task == 'hr':
            asyncio.run(self.run_hr)
        elif task == 'ecg':
            asyncio.run(self.run_ecg)

    def get(self):
        return self.data

    def cb_simple(self, sender, data):
        print(' '.join('{:02x}'.format(x) for x in data))

    def cb_hrm(self, sender, data):
        flag = data[0]  # 0x10 if has RR-interval, 0x00 otherwise
        bpm = data[1]
        self.data = bpm
        if self.data_only:
            return
        data = data[2:]
        has_rr = flag & 0x10
        if has_rr:
            n = int(len(data)/2)
            rrs = struct.unpack(f'<{n}H', data)
            rrs = np.array(rrs)/1024
            print(f'bpm: {bpm}, RR-interval (seconds): {rrs}, ref {1/np.mean(rrs)*60:.2f}')
        else:
            print(f'bpm: {bpm}')
    
    def cb_ecg(self, sender, data):
        tasktype = data[0]
        frametype = data[9]
        data = data[10:]
        assert len(data) % 3 == 0
        nd = int(len(data)/3)
        s1 = data.hex()
        s2 =  '00' + '00'.join([s1[i:i+6] for i in range(0, len(s1), 6)])
        s3 = bytes.fromhex(s2)
        d1 = struct.unpack(f'<{nd}i', s3)
        d1 = np.array(d1)
        d1 = np.right_shift(d1, 8)
        self.data = d1
        if self.data_only:
            return
            
        timestamp = struct.unpack('<Q', data[1:9])[0]/1e3   # in microseconds
        if self.sys_start_time is None:
            self.sys_start_time = datetime.datetime.now()
            self.sensor_start_time = timestamp

        current_time = self.sys_start_time + datetime.timedelta(microseconds=timestamp-self.sensor_start_time)
        if self.fig is None:
            plt.ion()
            self.fig, ax = plt.subplots()
            self.line, = ax.plot(np.arange(73), np.zeros(73))
            ax.set_xlim([0, 73])
            ax.set_ylim([-20000, 20000])
        print(current_time.strftime("%H:%M:%S:%f"))
        self.line.set_ydata(d1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    

    async def get_services(self, client):
        svcs = await client.get_services()
        print("Services:")
        for service in svcs:
            print(service)
        # Services:
        # 00001800-0000-1000-8000-00805f9b34fb (Handle: 1): Generic Access Profile
        # 00001801-0000-1000-8000-00805f9b34fb (Handle: 10): Generic Attribute Profile
        # 0000180d-0000-1000-8000-00805f9b34fb (Handle: 14): Heart Rate
        # 0000181c-0000-1000-8000-00805f9b34fb (Handle: 20): User Data
        # 0000180a-0000-1000-8000-00805f9b34fb (Handle: 44): Device Information
        # 0000180f-0000-1000-8000-00805f9b34fb (Handle: 59): Battery Service
        # 6217ff4b-fb31-1140-ad5a-a45545d7ecf3 (Handle: 63): Unknown
        # fb005c80-02e7-f387-1cad-8acd2d8df0c8 (Handle: 69): Unknown
        # 0000feee-0000-1000-8000-00805f9b34fb (Handle: 76): Polar Electro Oy

    async def get_info(self, client):
        device_name = await client.read_gatt_char(self.MODEL_NBR_UUID)
        print("Device Name:", device_name.decode())

        # model_number = await client.read_gatt_char(self.MODEL_NBR_UUID)
        # print("Model Number:", model_number.decode())

        # manufacturer_name = await client.read_gatt_char(self.MANUFACTURER_NAME_UUID)
        # print("Manufacturer Name:", manufacturer_name.decode())

        battery_level = await client.read_gatt_char(self.BATTERY_LEVEL_UUID)
        print("Battery Level:", battery_level[0])

    async def get_features(self, client):
        res = await client.read_gatt_char(self.PMD_CONTROL)
        assert res[0] == 15     # should always be 0x0f
        for k in self.FUNCTIONS:
            if (res[1] >> self.FUNCTIONS[k]) & 1 == 1:
                print('Support', k)

    async def get_settings(self, client, func='ECG'):
        def read_settings(sender, data):
            SETTINGS = ['sample rate', 'resolution', 'range']
            assert data[0] == 0xF0 and data[1] == 0x01      # should always be these
            if data[3] != 0:
                print('Error retrieving settings', data[3])
                return
            data = data[5:]
            while len(data) > 1:
                s = data[0]
                l = data[1]
                r = struct.unpack('<H', data[2:l*2+2])
                print('\t', SETTINGS[s], ':', r)
                data = data[l*2+2:]
        op = self.FUNCTIONS[func]
        print('Stream Setting of', func, ':')
        await client.start_notify(self.PMD_CONTROL, read_settings)
        await client.write_gatt_char(self.PMD_CONTROL, bytearray([0x01, op]), response=True)
        await client.stop_notify(self.PMD_CONTROL)
        # await client.start_notify(PMD_CONTROL, read_notification)

    async def start_streaming(self, client, cmd):
        def read_response(sender, data):
            assert data[0] == 0xF0 and data[1] == 0x02      # should always be these
            if data[3] != 0:
                print('Error start streaming', data[3])
            else:
                print('streaming started successfully')
        await client.start_notify(self.PMD_DATA, self.cb_ecg)
        await client.start_notify(self.PMD_CONTROL, read_response)
        await client.write_gatt_char(self.PMD_CONTROL, cmd, response=True)
        await client.stop_notify(self.PMD_CONTROL)
        while True:
            await asyncio.sleep(0.01)

    async def run_ecg(self):
        try:
            async with BleakClient(self.addr) as client:
                if not client.is_connected:        
                    print('Connection Failed')
                    return
                await self.get_info(client)
                # await self.get_services(client)
                # await self.get_features(client)
                await self.get_settings(client, 'ECG')
                await self.start_streaming(client, self.ECG_WRITE)
        except Exception as e:
            print(e)

    async def run_hr(self):
        try:
            async with BleakClient(self.addr) as client:
                if not client.is_connected:        
                    print('Connection Failed')
                    return
                await self.get_info(client)
                await client.start_notify(self.HR_MEASUREMENT, self.cb_hrm)
            
                while True:
                    await asyncio.sleep(0.1)
                # await client.write_gatt_char(self.HR_SERVICE_UUID, cmd, response=True)
                # await client.stop_notify(self.PMD_CONTROL)

                
        except Exception as e:
            print(e)

    

if __name__ == '__main__':
    ## This is the device MAC ID, please update with your device ID
    addr = "c1:02:a3:7c:57:42"
    h10 = H10(addr)

    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(h10.run())
    asyncio.run(h10.run_hr())