
import os
import sys
import time
import signal
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Recorder import Recorder
from DataLoader import DataLoader
from tensorflow.keras.models import load_model



g_recorder = None

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

def predict(model, recorder):
    recorder.window_export()
    plt.subplot(2, 1, 1)
    try:
        x = DataLoader.specgram(recorder.config['tepwav_file'], True)
    except Exception as e:
        print(e)
        plt.close()
        return 
    x = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    if x.shape != (1, 5511, 101):
        print('x shape error')
        plt.close()
        return
    predictions = model.predict(x)
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()

def detect(model, recorder):
    find = False
    recorder.window_export()
    plt.subplot(2, 1, 1)
    try:
        x = DataLoader.specgram(recorder.config['tepwav_file'], True)
    except Exception as e:
        print(e)
        plt.close()
        return
    x = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    if x.shape != (1, 5511, 101):
        print('x shape error')
        plt.close()
        return
    predictions = model.predict(x)
    print(predictions.shape)
    for i in range(1375):
        if predictions[0,i,0] > 0.5:
            find = True
    
    if find:
        plt.subplot(2, 1, 2)
        print('find')
        plt.plot(predictions[0,:,0])
        plt.ylabel('probability')
        plt.show()
    else:
        print('not find')
        plt.close()
    #os.remove(recorder.config['tepwav_file'])

def detect_from_wavfile(wavfile, model):
    if not model:
        model = load_model('md-normal-lr2.h5')
    plt.subplot(2, 1, 1)
    x = DataLoader.specgram(wavfile, True)
    x = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)

    plt.subplot(2, 1, 2)
    predictions = model.predict(x)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()

def get_model():
    gpu_setting()
    model = load_model('md-normal-lr2.h5')
    return model

def myexit():
    g_recorder.stop()
    exit()

if __name__ == '__main__':

    gpu_setting()

    model = load_model('md-normal-lr2.h5')

    if os.path.exists('Recorder-tmpwav.wav'):
        os.remove('Recorder-tmpwav.wav')

    rd = Recorder()
    
    rd.start()

    g_recorder = rd

    signal.signal(signal.SIGINT, myexit)
    signal.signal(signal.SIGTERM, myexit)

    #time.sleep(12)

    while True:
        try:
            key = input('\r\nPlease input, Enter:detect p:Predict k:Stop (Enter/p/k)?')
        except Exception as e:
            continue
        if key == 'k':
            myexit()
        if key == '':
            detect(model, rd)
        if key == 'p':
            predict(model, rd)
        #time.sleep(5)
