from __future__ import print_function

import argparse

from ctc_metrics import CtcMetrics
import cv2
from hyperparams import Hyperparams
import lstm
import mxnet as mx
import numpy as np
from ocr_iter import SimpleBatch

def read_img(path):
    img = cv2.resize(cv2.imread(path, 0), (80, 30)).astype(np.float32)/255
    img = np.expand_dims(img.transpose(1, 0), 0)
    
    return img

def lstm_init_states(batch_size):
    hp = Hyperparams()
    init_shapes = lstm.init_states(batch_size=batch_size, num_lstm_layer=hp.num_lstm_layer, num_hidden=hp.num_hidden)
    init_names = [s[0] for s in init_shapes]
    init_arrays = [mx.nd.zeros(x[1]) for x in init_shapes]
    
    return init_names, init_arrays


def load_module(prefix, epoch, data_names, data_shapes):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    
    pred_fc = sym.get_internals()['pred_fc_output']
    sym = mx.sym.softmax(data=pred_fc)
    
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data=data_names, label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes)
    mod.set_params(arg_params, aux_params, allowing_mossing=False)
    
    return mod

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the CAPTCHA image file")
    parser.add_argument("--prefix", help="Checkpoint prefix [Default 'ocr']", default='ocr')
    parser.add_argument("--epoch", help="Checkpoint epoch [Default 100]", type=int, default=100)
    args = parser.parse_args()

    init_state_names, init_state_arrays = lstm_init_states(batch_size=1)
    img = read_img(args.path)

    sample = SimpleBatch(
        data_names=['data'] + init_state_names,
        data=[mx.nd.array(img)] + init_state_arrays)

    mod = load_module(args.prefix, args.epoch, sample.data_names, sample.provide_data)
    
    mod.forward(sample)
    prob = mod.get_outputs()[0].asnumpy()
    
    prediction = CtcMetrics.ctc_label(np.argmax(prob, axis=-1).tolist())
    
    # Prediction
    prediction = [p-1 for p in prediction]
    print("Digits:", prediction)
    return

if __name__ == '__main__':
    main()
    

