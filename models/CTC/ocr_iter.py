"""
    Iterator for Captcha images used for LSTM-based OCR model.
"""

from __future__ import print_function

import numpy as np
import mxnet as mx 

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names=list(), label=list()):
        self._data = data
        self._label = label
        self._data_names = data_names
        self.label_names = label_names
        
        self.pad = 0
        self.index = None
        
    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label

    @property
    def data_names(self):
        return self._data_names

    @property
    def label_names(self):
        return self._label_names
    
    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self._data_names, self._data)]
    
    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self._label_names, self._label)]
    
def get_label(buf):
    ret = np.zeros(4)
    for in range(len(buf)):
        ret[i] = 1 + int(buf[i])
    if len(buf) == 3:
        ret[3] = 0
    return ret

class OCRIter(mx.io.DataIter):
    """
        Iterator class for generating captcha image data.
        
        Parameters:
        -----------
        count: int
            Number of batches to produce for one epoch.
        lstm_init_states: list of tuple(str, tuple)
            A list of tuples with [0] name and [1] shape of each lstm init state.
        captcha MPcaptcha
            Captcha image generator. Can be MPCaptcha or any other class providing. .shape and .get() interface.
    """
    def __init__(self, count, batch_size, lstm_init_states, captcha, name):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.count = count
        self.init_states = lstm_init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]
        data_shape = captcha.shape
        self.provide_data = [('data', (batch_size, data_shape[0], data_shape[1]))] + lstm_init_states
        self.provide_label = [('label', (self.batch_size, 4))]
        self.mp_captcha = captcha
        self.name = name
    
    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                img, num = self.mp_captcha.get()
                data.append(img)
                label.append(get_label(num))
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']
            
            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch
