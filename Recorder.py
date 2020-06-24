
import sys
import copy
import time
import wave
import pyaudio
import threading
import alsaaudio as aio
from scipy.io import wavfile



class RecorderAgent(threading.Thread):

    DEFAULT_CONFIG = {
        'rate':44100,
        'chunk':441,
        'channels':2,
        'format':aio.PCM_FORMAT_S16_LE,
        'window':10,
        'device':'hw:0,0',
    }

    def __init__(self, **configs):
        
        threading.Thread.__init__(self)

        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs.pop(key)

        self.stop_event = threading.Event()
        self.wait_event = threading.Event()
        self.wait_event.set()
        self.lock = threading.Lock()
        self.frames = []
        self.frames_length = int(self.config['rate']/self.config['chunk']*self.config['window'])

    def sleep(self):
        self.wait_event.clear()
        print('Recorder Agent sleep')

    def wakeup(self):
        self.wait_event.set()
        print('Recorder Agent wakeup')

    def stop(self):
        self.stop_event.set()
        print('Recorder Agent stop')
        
    def run(self):
        
        print('Recorder Agent start')

        self.pcmobj = aio.PCM(type=aio.PCM_CAPTURE, 
                              device=self.config['device'],
                              mode=aio.PCM_NORMAL)
        self.pcmobj.setchannels(self.config['channels'])
        self.pcmobj.setrate(self.config['rate'])
        self.pcmobj.setformat(self.config['format'])
        self.pcmobj.setperiodsize(self.config['chunk'])

        start = time.time()
        
        try:
            while not self.stop_event.is_set():
                self.wait_event.wait()
                self.run_once()
        except Exception as e:
            print('Record Agent Exception: {}'.format(e))

        self.pcmobj.close()
        print('time:{}'.format(int(time.time()-start)))

    def run_once(self):
        length, data = self.pcmobj.read()
        if len(self.frames) == self.frames_length:
            self.frames.pop(0)
        self.frames.append(data)

    def frames_export(self):
        self.sleep()
        if len(self.frames) == self.frames_length:
            frames = copy.deepcopy(self.frames)
        else:
            #print('data not ready')
            frames = None
        self.wakeup()
        return frames



class Recorder(object):

    DEFAULT_CONFIG = {
        'tepwav_file':'Recorder-tmpwav.wav',
    }

    def __init__(self, **configs):

        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs.pop(key)

        self.agent = RecorderAgent()

    @classmethod
    def record(cls, channels=2, fmt=aio.PCM_FORMAT_S16_LE, 
               rate=44100, periodsize=441, device='hw:0,0',
               seconds=10):

        frames = []

        pobj = aio.PCM(type=aio.PCM_CAPTURE, 
                       device='device', 
                       mode=aio.PCM_NORMAL)
        pobj.setchannels(channels)
        pobj.setrate(rate)
        pobj.setformat(fmt)
        pobj.setperiodsize(periodsize)

        pobj.dumpinfo()

        for i in range(0, int(rate/periodsize*seconds)):
            length, data = pobj.read()
            frames.append(data)

        pobj.close()

        return frames

    @classmethod
    def wavsave(cls, frames, channels=2, fmt=aio.PCM_FORMAT_S16_LE,
                rate=44100, filename='tmp.wav'):
        #print('frames length:{} framesize:{}'.format(
            #len(frames), len(frames[0])))
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(fmt)
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        sys.stdout.flush()

    def start(self):
        self.agent.setDaemon(True)
        self.agent.start()

    def stop(self):
        self.agent.stop()
        self.agent.join()

    def sleep(self):
        self.agent.sleep()
    
    def wakeup(self):
        self.agent.wakeup()

    def window_export(self):
        frames = self.agent.frames_export()
        if not frames:
            return
        Recorder.wavsave(frames, filename=self.config['tepwav_file'])