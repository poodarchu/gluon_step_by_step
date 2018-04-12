from __future__ import print_function

import argparse

import sys
import cv2
import numpy as np
import mxnet as mx
from collections import namedtuple
from ocr_iter import SimpleBatch
from captcha_generator import DigitCaptcha
from ctc_metrics import CtcMetrics
import lstm
from hyperparams import Hyperparams



class OCRLSTM(object):
    # Keep zero index for blank, CTC request it
    CONST_CHAR = '0123456789'
    
    def __init__(self, path_of_json, path_of_params):
        super(OCRLSTM, self).__init__()
        self.path_of_json = path_to_json
        self.path_of_params = path_of_params
        self.predictor = None
        self.__init_ocr()
        
    def __init_ocr(self):
        num_label = 4 # Set your max length of label, add one more for blank
        batch_size = 1

        num_hidden = 100
        num_lstm_layer = 2
        
        init_c = [('l%d_init_c' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_h = [('l%d_init_h' % l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
        init_states = init_c + init_h
        
        init_state_arrays = np.zeros((batch_size, num_hidden), dtype='float32')
        self.init_state_dict = {}
        
        for x in init_states:
            self.init_state_dict[x[0]] = init_state_arrays
            
        all_shapes = [('data', (batch_size, 80, 30))] + init_states + [('label', (batch_size, num_label))]
        all_shapes_dict = {}
        for _shape in all_shapes:
            all_shapes_dict[_shape[0]] = _shape[1]
        self.predictor = Predictor(open(self.path_of_json, 'rb').read(),
                                   open(self.path_of_params, 'rb').read(),
                                   all_shapes_dict)

    def forward_ocr(self, img_):
        img_ = cv2.resize(img_, (80, 30))
        img_ = img_.transpose(1, 0)
        print(img_.shape)
        img_ = img_.reshape((1, 80, 30))
        print(img_.shape)
        # img_ = img_.reshape((80 * 30))
        img_ = np.multiply(img_, 1 / 255.0)
        self.predictor.forward(data=img_, **self.init_state_dict)
        prob = self.predictor.get_output(0)
        label_list = []
        for p in prob:
            print(np.argsort(p))
            max_index = np.argsort(p)[::-1][0]
            label_list.append(max_index)
        return self.__get_string(label_list)

    @staticmethod
    def __get_string(label_list):
        # Do CTC label rule
        # CTC cannot emit a repeated symbol on consecutive timesteps
        ret = []
        label_list2 = [0] + list(label_list)
        for i in range(len(label_list)):
            c1 = label_list2[i]
            c2 = label_list2[i+1]
            if c2 == 0 or c2 == c1:
                continue
            ret.append(c2)
        # change to ascii
        s = ''
        for l in ret:
            if l > 0 and l < (len(lstm_ocr_model.CONST_CHAR)+1):
                c = lstm_ocr_model.CONST_CHAR[l-1]
            else:
                c = ''
            s += c
        return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("predict_lib_path", help="Path to directory containing mxnet_predict.so")
    args = parser.parse_args()

    sys.path.append(args.predict_lib_path + "/python")
    from mxnet_predict import Predictor

    _lstm_ocr_model = lstm_ocr_model('ocr-symbol.json', 'ocr-0010.params')
    img = cv2.imread('sample0.png', 0)
    _str = _lstm_ocr_model.forward_ocr(img)
    print('Result: ', _str)
                                                                                  