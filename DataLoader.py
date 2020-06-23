
import os
import copy
from scipy.io import wavfile
from pydub import AudioSegment


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
        if not os.path.exists(self.config['gen_path'])
            return False
        return True

    def info(self):

        string = '-- DataLoader info --\r\n'
        
        if not self.is_loader():
            string += 'not load\r\n'
        else:
            string += 'loaded from {}\r\n'.format(self.config['ori_path'])
            string += '\tactivate:{}\r\n'.format(len(self.ori_activate))
            string += '\tnegative:{}\r\n'.format(len(self.ori_negative))
            string += '\tbackground:{}\r\n'.format(len(self.ori_background))

        if not self.is_generated():
            string += 'not generate\r\n'
        else:
            string += 'generated to {}\r\n'.format(self.config['gen_path'])
            if not os.path.exists(self.gen_train_path)
                string += 'no train\r\n'
            else:
                string += 'generated train to {}\r\n'.format(self.gen_train_path)
                string += '\tbatchs:{} \r\n'.format(self.gen_train_batchs)
                string += '\tbatchsize:{} \r\n'.format(self.gen_train_batch_size)
            if not os.path.exists(self.gen_dev_path)
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

        print('oridata loaded from {}, activate:{} negativa:{} background:{}'.format(
            oripath,
            len(self.ori_activate),
            len(self.ori_negavite),
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
                random_nr = np.random.randint(0, len(size))
            else:
                random_nr = size
        random_indexs = np.random.randint(len(datas), size=random_nr)
        random_datas = [datas[i] for i in random_indexs]
        return random_datas

    def gen_clean(self, target_path):
        for root, dirs, files in os.walk(target_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def gen_mkdir(self, target_path):
        dataset_path = os.path.join(target_path, self.config['gen_dataset_dir'])
        synthesis_path = os.path.join(target_path, self.config['gen_synthesis_dir'])
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        if not os.path.exists(synthesis_path):
            os.mkdir(synthesis_path)

    def _generator(self, backgrounds, batch, count):
        
        for bg in backgrounds:

            y = np.zeros((1, self.config['ywidth']))
            bg -= 20
            inserted_segment = []

            random_activates = self.get_random_data(self.ori_activate, 5)
            for activate in random_activates:
                bg, time = self.insert_audio_clip(bg, activate, inserted_segment)
                if time:
                    start, end = time
                    y = self.insert_ones(y, segmet_end_ms=end)

            random_negatives = self.get_random_data(self.ori_negative, 3)
            for negative in random_negatives:
                bg, _ = self.insert_audio_clip(bg, negative, inserted_segment)
            
            bg = self.match_target_amplitude(bg, -20.0)

            savefile = '{}/synthesis-{}-{}.wav'.format(self.synthesis_path, batchs, count)
            bg.export(savefile, format='wav')
            x = DataLoad.specgram(savefile)
            yield (x,y)

    def generate(self, dir=None, target='train', batchs=1, batch_size=100):

        target_dir = None
        target_path = None
        count = 0

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
            target_path = os.path.join(self.config['gen_dir'],self.config['gen_train_dir'])
        elif target == 'dev':
            target_path = os.path.join(self.config['gen_dir'],self.config['gen_dev_dir'])

        self.dataset_path = os.path.join(target_path, self.config['gen_dataset_dir'])
        self.synthesis_path = os.path.join(target_path, self.config['gen_synthesis_dir'])

        self.gen_clean(target_path)
        self.gen_mkdir(target_path)

        self.gen_X = np.zeros((batch_size, 5511, 101))
        self.gen_Y = np.zeros((batch_size, 1375, 1))

        try:
            for i in range(batchs):

                count = 0
                random_backgrounds = self.get_random_data(self.ori_background, 
                                                          size=batch_size,
                                                          fixed_size=True)

                for x, y in slef._generator(random_backgrounds, i, count):
                    self.gen_X[count] = x.transpose()
                    self.gen_Y[count] = y.transpose()
                    count += 1
                
                np.save('{}/X-{}.npy'.format(self.dataset_path, i), self.gen_X)
                np.save('{}/Y-{}.npy'.format(self.dataset_path, i), self.gen_Y)
                print('generated dataset batch{} to {}'.format(self.dataset_path))
        except Exception as e:
            print(e)
            return
