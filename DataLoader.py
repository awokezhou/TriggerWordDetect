
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
from pydub.playback import play 



class DataLoader(object):

    DEFAULT_CONFIG = {
        'ori_dir':'ori',
        'gen_dir':'gen',
        'gen_train_dir':'train',
        'gen_dev_dir':'dev',
        'gen_dataset_dir':'dataset',
        'gen_synthesis_dir':'synthesis',
        'labels':['activate','negative','background'],
        'ywidth':1375,
        'gen_batchs':1,
        'gen_batch_size':100,
    }

    def __init__(self, **configs):

        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs.pop(key)

        self.ori_activate = []
        self.ori_negative = []
        self.ori_background = []

        self.gen_train_path = os.path.join(self.config['gen_dir'],self.config['gen_train_dir'])
        self.gen_dev_path = os.path.join(self.config['gen_dir'],self.config['gen_dev_dir'])
        self.gen_train_batchs = self.config['gen_batchs']
        self.gen_train_batch_size = self.config['gen_batch_size']
        self.gen_dev_batchs = self.config['gen_batchs']
        self.gen_dev_batch_size = self.config['gen_batch_size']

    def is_loaded(self):
        if len(self.ori_activate) == 0 and \
           len(self.ori_negative) == 0 and \
           len(self.ori_background) == 0:
           return False
        return True

    def is_generated(self):
        if not os.path.exists(self.config['gen_dir']):
            return False
        return True

    def info(self):

        string = '-- DataLoader info --\r\n'
        
        if not self.is_loaded():
            string += 'not load\r\n'
        else:
            string += 'loaded from {}\r\n'.format(self.config['ori_dir'])
            string += '\tactivate:{}\r\n'.format(len(self.ori_activate))
            string += '\tnegative:{}\r\n'.format(len(self.ori_negative))
            string += '\tbackground:{}\r\n'.format(len(self.ori_background))

        if not self.is_generated():
            string += 'not generate\r\n'
        else:
            string += 'generated to {}\r\n'.format(self.config['gen_dir'])
            if not os.path.exists(self.gen_train_path):
                string += 'no train\r\n'
            else:
                string += 'generated train to {}\r\n'.format(self.gen_train_path)
                string += '\tbatchs:{} \r\n'.format(self.gen_train_batchs)
                string += '\tbatchsize:{} \r\n'.format(self.gen_train_batch_size)
            if not os.path.exists(self.gen_dev_path):
                string += 'no dev\r\n'
            else:
                string += 'generated dev to {}\r\n'.format(self.gen_dev_path)
                string += '\tbatchs:{} \r\n'.format(self.gen_dev_batchs)
                string += '\tbatchsize:{} \r\n'.format(self.gen_dev_batch_size)
        print(string)

    def load(self, path=None):

        oripath = None
        valnames = self.__dict__

        if not path:
            oripath = self.config['ori_dir']
        else:
            oripath = path
            self.config['ori_dir'] = path

        for label in self.config['labels']:
            dirname = os.path.join(oripath, label)
            for file in os.listdir(dirname):
                if file[-4:] == '.wav':
                    data = AudioSegment.from_wav(os.path.join(dirname, file))
                    valnames['ori_'+label].append(data)

        print('data loaded from {}, activate:{} negativa:{} background:{}'.format(
            oripath,
            len(self.ori_activate),
            len(self.ori_negative),
            len(self.ori_background)
        ))

    @classmethod
    def specgram(self, file, show=False):
        rate, data = wavfile.read(file)
        nfft = 200
        fs = 8000
        noverlap = 120
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
        else:
            pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap=noverlap)
        if not show:
            plt.close()
        return pxx

    def get_random_segment(self, segment_ms):
        start = np.random.randint(low=0, high=10000-segment_ms)
        end = start + segment_ms - 1
        return (start, end)

    def get_random_data(self, datas, size=0, fixed_size=False):
        if size == 0:
            random_nr = np.random.randint(0, len(datas))
        else:
            if not fixed_size:
                random_nr = np.random.randint(0, size)
            else:
                random_nr = size
        random_indexs = np.random.randint(len(datas), size=random_nr)
        random_datas = [datas[i] for i in random_indexs]
        return random_datas

    def match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def insert_ones(self, y, segment_end_ms):
        segment_end_y = int(segment_end_ms * self.config['ywidth'] / 10000.0)
        #print('insert y:{}'.format(segment_end_y))
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < self.config['ywidth']:
                y[0, i] = 1
        return y

    def is_overlapping(self, segment_time, inserted_segment):
        start, end = segment_time
    
        ### START CODE HERE ### (≈ 4 line)
        # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
        overlap = False
        
        # Step 2: loop over the previous_segments start and end times.
        # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
        for inserted_start, inserted_end in inserted_segment:
            if start <= inserted_end and end >= inserted_start:
                overlap = True
                break
        ### END CODE HERE ###

        return overlap    


    def insert_audio_clip(self, background, clip, inserted_segment):
       
        inserted = True
        #print('inserted:{}'.format(inserted_segment))

        segment_ms = len(clip)
        #print('segm ms:{}'.format(segment_ms))
        segment_time = self.get_random_segment(segment_ms)
        #print('segm time:{}'.format(segment_time))

        i = 0
        while self.is_overlapping(segment_time, inserted_segment):
            segment_time = self.get_random_segment(segment_ms)
            #print("reset segm time:{}".format(segment_time))
            i += 1
            time.sleep(0.1)
            if i >= 10:
                inserted = False
                break

        if inserted:
            #print('segm insert:{}'.format(segment_time))
            inserted_segment.append(segment_time)
            new_background = background.overlay(clip, position = segment_time[0])
            return new_background, segment_time
        else:
            return background, None

    def dir_clean(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def gen_clean(self, target_path=None):
        path = None
        if not target_path:
            path = self.config['gen_dir']
        else:
            path = target_path
        self.dir_clean(path)
        print('{} clean'.format(path))

    def gen_mkdir(self, target_path):
        dataset_path = os.path.join(target_path, self.config['gen_dataset_dir'])
        synthesis_path = os.path.join(target_path, self.config['gen_synthesis_dir'])
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        if not os.path.exists(synthesis_path):
            os.mkdir(synthesis_path)
        print('{} mkdir'.format(target_path))

    def gen_check(self, dir=None, target='train', batchs=0, index=0):
        target_dir = None
        target_path = None

        if not self.is_generated():
            print('not generated')
            return
        
        if not dir:
            target_dir = self.config['gen_dir']
        else:
            target_dir = dir

        if target == 'train':
            target_path = os.path.join(target_dir,self.config['gen_train_dir'])
        elif target == 'dev':
            target_path = os.path.join(target_dir,self.config['gen_dev_dir'])

        dataset_path = os.path.join(target_path, self.config['gen_dataset_dir'])
        synthesis_path = os.path.join(target_path, self.config['gen_synthesis_dir'])

        Y = np.load('{}/Y-{}.npy'.format(dataset_path, batchs))
        y = Y[index]
        
        filename = '{}/synthesis-{}-{}.wav'.format(
            synthesis_path, batchs, index)
        
        print('please run follow command to play wavfile:\r\n')
        print('\taplay -i {}'.format(filename))

        plt.subplot(2, 1, 1)
        x = DataLoader.specgram(filename, True)
        x  = x.swapaxes(0,1)
        x = np.expand_dims(x, axis=0)
        
        plt.subplot(2, 1, 2)
        plt.plot(y)
        plt.show()

    def _generator(self, backgrounds, batch):
        
        for bg in backgrounds:

            y = np.zeros((1, self.config['ywidth']))
            bg -= 20
            inserted_segment = []

            random_activates = self.get_random_data(self.ori_activate, 5)
            #print('random activates number:{}'.format(len(random_activates)))
            for activate in random_activates:
                bg, time = self.insert_audio_clip(bg, activate, inserted_segment)
                if time:
                    start, end = time
                    y = self.insert_ones(y, segment_end_ms=end)

            random_negatives = self.get_random_data(self.ori_negative, 3)
            #print('random negatives number:{}'.format(len(random_negatives)))
            for negative in random_negatives:
                bg, _ = self.insert_audio_clip(bg, negative, inserted_segment)
            
            bg = self.match_target_amplitude(bg, -20.0)

            savefile = '{}/synthesis-{}-{}.wav'.format(
                self.synthesis_path, batch, self.synthesis_count)
            bg.export(savefile, format='wav')
            print('synthesis:{} exportd'.format(savefile))
            x = DataLoader.specgram(savefile)
            yield (x,y)

    def generate(self, dir=None, target='train', batchs=1, batch_size=100):

        target_dir = None
        target_path = None
        self.synthesis_count = 0

        np.random.seed(22)

        if not self.is_loaded():
            print('not loaded')
            return

        if not dir:
            target_dir = self.config['gen_dir']
        else:
            target_dir = dir
            self.config['gen_dir'] = dir

        if target == 'train':
            target_path = os.path.join(target_dir,self.config['gen_train_dir'])
        elif target == 'dev':
            target_path = os.path.join(target_dir,self.config['gen_dev_dir'])

        self.dataset_path = os.path.join(target_path, self.config['gen_dataset_dir'])
        self.synthesis_path = os.path.join(target_path, self.config['gen_synthesis_dir'])

        print('target:{}'.format(target))
        print('target dir:{} path:{}'.format(target_dir, target_path))
        print('dataset path:{}'.format(self.dataset_path))
        print('synthesis path:{}'.format(self.synthesis_path))

        self.gen_clean(target_path)
        self.gen_mkdir(target_path)

        self.gen_X = np.zeros((batch_size, 5511, 101))
        self.gen_Y = np.zeros((batch_size, 1375, 1))

        try:
            for i in range(batchs):

                self.synthesis_count = 0
                random_backgrounds = self.get_random_data(self.ori_background, 
                                                          size=batch_size,
                                                          fixed_size=True)
                #print('random backgrounds number {}'.format(len(random_backgrounds)))
                for x, y in self._generator(random_backgrounds, i):
                    self.gen_X[self.synthesis_count] = x.transpose()
                    self.gen_Y[self.synthesis_count] = y.transpose()
                    self.synthesis_count += 1
                
                np.save('{}/X-{}.npy'.format(self.dataset_path, i), self.gen_X)
                np.save('{}/Y-{}.npy'.format(self.dataset_path, i), self.gen_Y)
                print('generated dataset batch{} to {}'.format(
                    i, self.dataset_path))
        except Exception as e:
            print(e)
            return
