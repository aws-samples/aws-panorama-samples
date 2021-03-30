import argparse
import logging
import os
import json
import time

import numpy as np

import mxnet as mx
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from mxnet import gluon, autograd

INPUT_SIZE = 512
MODEL = 'ssd_512_mobilenet1.0_custom'
CLASSES = ['pikachu']

def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader

def get_ctx():
    try:
        ctx = mx.gpu(0)
    except Exception as e:
        ctx = mx.cpu()
    print('Using {}'.format(ctx))
    return ctx
    
def get_net(model_name=MODEL, classes=CLASSES, transfer='voc', ctx=mx.cpu()):
    print(f'Loading model [{model_name}] for [{transfer}] transfer training from Model Zoo')
    net = get_model(model_name, classes=classes, pretrained_base=False, transfer=transfer, ctx=ctx)
    return net
        
def train(args):

    epochs = args.epochs
    model_dir = args.model_dir
    train_dir = args.train

    # Load the model into CPU first
    net = get_net()
    
    rec_file_path = os.path.join(train_dir, 'train.rec')
    print(f'--> Create a dataset from {rec_file_path}')
    dataset = gcv.data.RecordFileDetection(rec_file_path)
    
    print('--> Create a data loader')
    train_data = get_dataloader(net, dataset, 512, 16, 0)
    
    ctx = get_ctx()
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})
    
    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    print('--> Start training')
    for epoch in range(epochs):
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=[ctx], batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=[ctx], batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=[ctx], batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since the loss is already normalized, no more normalization is needed. 
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            if i % 20 == 0:
                print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()
            
    print('Finished training, export the trained model')

    export_prefix = '{}/{}'.format(model_dir, args.export_prefix)

    net.hybridize()
    input_shape = [1, 3, INPUT_SIZE, INPUT_SIZE]
    net(mx.ndarray.zeros(input_shape, ctx=ctx))

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    net.export(export_prefix)
    print(f'Exported the model under prefix {export_prefix}***')
      
    if args.save_params:
        parameters_file = '{}/{}'.format(model_dir, args.saved_params_file)
        net.save_parameters(parameters_file)
        print(f'Saved the model parameters to {parameters_file}')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    model_dir = os.environ['SM_MODEL_DIR'] if 'SM_MODEL_DIR' in os.environ else 'model'
    train = os.environ['SM_CHANNEL_TRAINING'] if 'SM_CHANNEL_TRAINING' in os.environ else 'train'
    parser.add_argument('--model-dir', type=str, default=model_dir)
    parser.add_argument('--train', type=str, default=train)
    parser.add_argument('--export-prefix', type=str, default='exported-model')
    parser.add_argument('--save-params', action='store_true')
    parser.add_argument('--saved-params-file', type=str, default='saved-model.params')
    
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    args = parse_args()        
    train(args)