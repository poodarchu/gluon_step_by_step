from __future__ import print_function

import argparse
import logging
import os

from captcha_generator import MPDigitCaptcha
from hyperparams import Hyperparams
from ctc_metrics import CtcMetrics
import lstm
import mxnet as mx
from oct_iter import OCTIter

def get_fonts(path):
    fonts = list()
    if os.path.isdir(path):
        for filename in os.path.listdir(path):
            if filename.endswith('.ttf'):
                fonts.append(os.path.join(path, filename))
                
    return fonts

def parse_args():
    """
        Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("font_path", help="Path to ttf font file or directory containing ttf files")
    parser.add_argument("--loss", help="'ctc' or 'warpctc' loss", default='ctc')
    parser.add_argument("--cpu", help="Num of CPUs for training", type=int)
    parser.add_argument("--num_proc", help="number CAPTCHA generating process", type=int, default=4)
    parser.add_argument("--prefix", help="ckpt prefix", default='ocr')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not any(args.loss == s for s in ['ctc', 'warpctc']):
        raise ValueError("Invalid loss {}".format(args.loss))
        
    mp_captcha = MPDigitCaptcha(
        font_paths=get_fonts(args.font_path), h=hp.seq_length, w=30,
        num_digit_min=3, num_digit_max=4, num_processes=args.num_proc, max_queue_size=hp.batch_size*2)
    
    try:
        mp_captcha.strat()
        
        if args.gpu:
            contexts = [mx.context.gpu(i) for i in range(args.gpu)]
        else:
            contexts = [mx.context.cpu(i) for i in range(args.cpu)]
            
        init_states = lstm.init_states(hp.batch_size, hp.num_lstm_layer, hp.num_hidden)
        
        data_train = OCRIter(
            hp.train_epoch_size // hp.batch_size, hp.batch_size, init_states, captcha=mp_captcha, name='train')
        data_eval = OCRIter(
            hp.eval_epoch_size // hp.batch_size, hp.batch_size, init_states, captcha=mp_captcha, name='eval')
        
        symbol = lstm.unroll(
            num_lstm_layer=hp.num_lstm_layer,
            seq_len=seq_length,
            num_hidden=hp.num_hidden,
            num_label=hp.num_label,
            loss_type=args.loss)
        
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=head)
        
        module = mx.mod.Module(
            symbol,
            data_names=['data', 'l0_init_c', 'l0_init_h', 'l1_init_c', 'l1_init_h'],
            label_names=['label'],
            context=contexts)
        
        metrics = CtcMetrics(hp.seq_length)
        module.fit(train_data=data_train,
                   eval_data=data_eval,
                   eval_metric=mx.metric.np(metrics.accuracy, allow_extra_outputs=True),
                   optimizer='sgd'
                   optimizer_params={
                       'learning_rate':hp.learning_rate,
                       'momentum':hp.momentum,
                       'wd':0.0001,
                   },
                   initializer=mx.init.Xavier(factor_type='in', magnitude=2.34),
                   num_epoch=hp.num_epoch,
                   batch_end_callback=mx.callback.Speedometer(hp.batch_size, 50),
                   epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
                  )
        
    except KeyboardInterrupt:
        print("W: interrupt received, stopping...")
    finally:
        mp_captcha.reset()

if __name__ = '__main__':
    main()
    