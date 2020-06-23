
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Recorder import Recorder
from DataLoader import DataLoader
from tensorflow.keras.models import load_model


# from tensorflow.keras.models import load_model
# from detect import gpu_setting, detect
# gpu_setting()
# model = load_model('md-normal-lr2.h5')
# from Recorder import Recorder
# rd = Recorder()
# rd.start()
# detect()

def gpu_setting(limit=2048):
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for _g in gpus:
        tf.config.experimental.set_memory_growth(_g,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
        tf.config.experimental.set_memory_growth(_g, True)

def detect(model, recorder):
    recorder.sleep()
    file= recorder.window_export()
    plt.subplot(2, 1, 1)
    x = DataLoader.specgram(file, True)
    x = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    recorder.wakeup()

def detect_from_wavfile(wavfile):
    gpu_setting()
    model = load_model('md-normal-lr2.h5')
    plt.subplot(2, 1, 1)
    x = DataLoader.specgram(wavfile, True)
    x = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()

def test():
    gpu_setting()
    model = load_model('md-normal-lr2.h5')
    rd = Recorder()
    rd.start()
    return (model, rd)

if __name__ == '__main__':

    model = load_model('md-normal-lr2.h5')

    rd = Recorder()
    rd.start()

    while True:
        detect(model, rd)
