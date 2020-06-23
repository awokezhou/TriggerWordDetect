
import copy
import wave
import pyaudio
import threading
from scipy.io import wavfile



class RecorderAgent(threading.Thread):

    DEFAULT_CONFIG = {
        'rate':44100,
        'chunk':44100,
        'channels':1,
        'format':pyaudio.paInt16,
        'window':10,
        'tepwav_file':'Recorder-tmpwav.wav',
    }

    def __init__(self, **configs):
        
        threading.Thread = __init__(self)

        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs.pop(key)

        self.stop_event = threading.Event()
        self.audio = pyaudio.PyAudio()
        self.frames = []

    def stop(self):
        
        self.stop_event.set()
        print('Recorder Agent stop')
        
    def run(self):
        
        print('Recorder Agent start')

        try:
            stream = self.audio.open(format=self.config['format'],
                                    channels=self.config['channels'],
                                    rate=self.config['rate'],
                                    frames_per_buffer=self.config['chunk'],
                                    input=True)
        except Exception as e:
            print('Record Agent Exception: {}'.format(e))
            return
        
        print('stream open')

        try:
            while not self.stop_event.is_set():
                self.run_once(stream, self.config['chunk'])
        except Exception as e:
            print('Record Agent Exception: {}'.format(e))

        stream.stop_stream()
        stream.close()
        self.audio.terminate()

        print('stream close')

    def run_once(self, stream, chunk):
        
        data = stream.read(stream, chunk)
        
        if len(self.frames) == self.config['window']:
            self.frames.pop(0)
        
        self.frames.append(data)

    def wav_extract(self):
        frames = copy.deepcopy(self.frames)
        wf = wave.open(self.config['tepwav_file'], 'wb')
        wf.setnchannels(self.config['channels'])
        wf.setsampwidth(self.audio.get_sample_size(self.agent.config['format']))
        wf.setframerate(self.config['rate'])
        wf.writeframes(b''.join(frames))
        wf.close()
        rate, data = wavfile.read(self.config['tepwav_file'])
        return data



class Recorder(object):

    DEFAULT_CONFIG = {
        'record_rate':44100,
        'record_chunk':44100,
        'record_channels':1,
        'record_format':pyaudio.paInt16,
    }

    def __init__(self, **configs):

        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs.pop(key)

        self.agent = RecorderAgent(rate=self.config['record_rate'],
                                   chunk=self.config['record_chunk'],
                                   format=self.config['record_format'],
                                   channels=self.config['record_channels'],
                                   window=10)

    def record_seconds(self, sec, save=False, savefile='Recorder-test.wav'):
        
        frames = []
        p = pyaudio.PyAudio()
        
        stream = p.open(format=self.config['record_format'],
                        channels=self.config['record_channels'],
                        rate=self.config['record_rate'],
                        frames_per_buffer=self.config['record_chunk'],
                        input=True)
        
        max = int(self.config['record_rate']/self.config['record_chunk']*sec)
        for i in range(0, max):
            data = stream.read(self.config['record_chunk'])
            frames.append(data)
        
        stream.stop_stream()
        stream.close()

        if save:
            wf = wave.open(savefile, 'wb')
            wf.setnchannels(self.config['record_channels'])
            wf.setsampwidth(p.get_sample_size(self.config['record_format']))
            wf.setframerate(self.config['record_rate'])
            wf.writeframes(b''.join(frames))
            wf.close()

        p.terminate()

    def start(self):
        self.agent.start()

    def stop(self):
        self.agent.stop()
        self.agent.join()

    def window_extract(self):
        return self.agent.wav_extract()

