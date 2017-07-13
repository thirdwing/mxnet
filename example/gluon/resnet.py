from __future__ import division, print_function

import argparse, time
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag

from data import *

# CLI
parser = argparse.ArgumentParser(description='Train a resnet model for image classification.')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset to use. options are mnist, cifar10, and dummy.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--resnet-version', type=int, default=1,
                    help='whether to use ResnetV1 or ResnetV2. default is 1.')
parser.add_argument('--resnet-layers', type=int, default=50,
                    help='layers of resnet to use. options are 18, 50. default is 50.')
parser.add_argument('--gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning Rate. default is 0.01.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123.')
parser.add_argument('--thumbnail', action='store_true', default=False,
                    help='use thumbnail or not. default is false.')
parser.add_argument('--benchmark', action='store_true', default=False,
                    help='whether to run benchmark.')
parser.add_argument('--symbolic', action='store_true', default=False,
                    help='whether to train in symbolic way with module.')
parser.add_argument('--log-interval', type=int, default=100,
                    help='Number of batches to wait before logging.')
opt = parser.parse_args()

print(opt)


# Define network

def conv3x3(filters, stride, in_channels):
    return nn.Conv2D(filters, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

class BasicBlockV1(gluon.HybridBlock):
    def __init__(self, filters, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3(filters, stride, in_channels)
            self.bn1 = nn.BatchNorm(in_channels=in_channels)
            self.conv2 = conv3x3(filters, 1, filters)
            self.bn2 = nn.BatchNorm(in_channels=filters)
            if downsample:
                self.conv_ds = nn.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False, in_channels=in_channels)
                self.bn_ds = nn.BatchNorm(in_channels=filters)
            self.downsample = downsample

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.Activation(x, act_type='relu')

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)

        out = residual + out
        out = F.Activation(out, act_type='relu')

        return out


class BottleneckV1(gluon.HybridBlock):
    def __init__(self, filters, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(filters//4, kernel_size=1, strides=1, in_channels=in_channels)
            self.bn1 = nn.BatchNorm(in_channels=filters//4)
            self.conv2 = conv3x3(filters//4, stride, filters//4)
            self.bn2 = nn.BatchNorm(in_channels=filters//4)
            self.conv3 = nn.Conv2D(filters, kernel_size=1, strides=1, in_channels=filters//4)
            self.bn3 = nn.BatchNorm(in_channels=filters)
            if downsample:
                self.conv_ds = nn.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False, in_channels=in_channels)
                self.bn_ds = nn.BatchNorm(in_channels=filters)
            self.downsample = downsample

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.Activation(out, act_type='relu')

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.Activation(out, act_type='relu')

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)

        out = out + residual

        out = F.Activation(out, act_type='relu')
        return out


class ResnetV1(gluon.HybridBlock):
    def __init__(self, block, classes, layers, filters, thumbnail=False, **kwargs):
        super(ResnetV1, self).__init__(**kwargs)
        with self.name_scope():
             assert len(layers) == len(filters) - 1
             self._thumbnail = thumbnail
             if thumbnail:
                 self.conv0 = conv3x3(filters[0], 1, 3)
             else:
                 self.conv0 = nn.Conv2D(filters[0], 7, 2, 3, use_bias=False,
                                        in_channels=3)
                 self.bn0 = nn.BatchNorm(in_channels=filters[0])
                 self.pool0 = nn.MaxPool2D(3, 2, 1)

             self.body = nn.HSequential()
             in_channels = filters[0]
             for i in range(len(layers)):
                 stride = 1 if i == 0 else 2
                 self.body.add(self._make_layer(block, layers[i], filters[i+1],
                                                stride, in_channels=filters[i]))
                 in_channels = filters[i+1]

             self.pool1 = nn.GlobalAvgPool2D()
             self.dense1 = nn.Dense(classes, in_units=filters[-1])

    def _make_layer(self, block, layers, filters, stride, in_channels=0):
        layer = nn.HSequential()
        layer.add(block(filters, stride, True, in_channels=in_channels))
        for i in range(layers-1):
            layer.add(block(filters, 1, False, in_channels=filters))
        return layer

    def hybrid_forward(self, F, x):
        x = self.conv0(x)
        if not self._thumbnail:
            x = self.bn0(x)
            x = F.Activation(x, act_type='relu')
            x = self.pool0(x)

        x = self.body(x)

        x = self.pool1(x)
        x = x.reshape((0, -1))
        x = self.dense1(x)

        return x


class BasicBlockV2(gluon.HybridBlock):
    def __init__(self, filters, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        with self.name_scope():
            self.bn1 = nn.BatchNorm(in_channels=in_channels)
            self.conv1 = conv3x3(filters, stride, in_channels)
            self.bn2 = nn.BatchNorm(in_channels=filters)
            self.conv2 = conv3x3(filters, 1, filters)
            if downsample:
                self.downsample = nn.Conv2D(filters, 1, stride, use_bias=False,
                                            in_channels=in_channels)
            else:
                self.downsample = None

    def hybrid_forward(self, F, x):
        if not self.downsample:
            residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class BottleneckV2(gluon.HybridBlock):
    def __init__(self, filters, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        with self.name_scope():
            self.bn1 = nn.BatchNorm(in_channels=in_channels)
            self.conv1 = conv3x3(filters//4, 1, in_channels)
            self.bn2 = nn.BatchNorm(in_channels=filters//4)
            self.conv2 = conv3x3(filters//4, stride, filters//4)
            self.bn3 = nn.BatchNorm(in_channels=filters//4)
            self.conv3 = conv3x3(filters, 1, filters//4)
            if downsample:
                self.downsample = nn.Conv2D(filters, 1, stride, use_bias=False,
                                            in_channels=in_channels)
            else:
                self.downsample = None

    def hybrid_forward(self, F, x):
        if not self.downsample:
            residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual

class ResnetV2(gluon.HybridBlock):
    def __init__(self, block, classes, layers, filters, thumbnail=False, **kwargs):
        super(ResnetV2, self).__init__(**kwargs)
        with self.name_scope():
            assert len(layers) == len(filters) - 1
            self._thumbnail = thumbnail
            self.bn_data = nn.BatchNorm(in_channels=3, scale=False, center=False)
            if thumbnail:
                self.conv0 = conv3x3(filters[0], 1, 3)
            else:
                self.conv0 = nn.Conv2D(filters[0], 7, 2, 3, use_bias=False,
                                       in_channels=3)
                self.bn0 = nn.BatchNorm(in_channels=filters[0])
                self.pool0 = nn.MaxPool2D(3, 2, 1)

            self.body = nn.HSequential()
            in_channels = filters[0]
            for i in range(len(layers)):
                stride = 1 if i == 0 else 2
                self.body.add(self._make_layer(block, layers[i], filters[i+1],
                                               stride, in_channels=in_channels))
                in_channels = filters[i+1]

            self.bn1 = nn.BatchNorm(in_channels=in_channels)
            self.pool1 = nn.GlobalAvgPool2D()
            self.dense1 = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, filters, stride, in_channels=0):
        layer = nn.HSequential()
        layer.add(block(filters, stride, True, in_channels=in_channels))
        for i in range(layers-1):
            layer.add(block(filters, 1, False, in_channels=filters))
        return layer

    def hybrid_forward(self, F, x):
        x = self.bn_data(x)
        x = self.conv0(x)
        if not self._thumbnail:
            x = self.bn0(x)
            x = F.Activation(x, act_type='relu')
            x = self.pool0(x)

        x = self.body(x)

        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        x = self.pool1(x)
        x = x.reshape((0, -1))
        x = self.dense1(x)

        return x

# construct net
resnet_spec = { 18: ('basic_block', [2, 2, 2], [16, 16, 32, 64]),
                34: ('basic_block', [3, 4, 6, 3], [16, 16, 32, 64]),
                50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
                101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
                152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048]) }

resnet_net_versions = [ResnetV1, ResnetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                  {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]

def get_resnet(version, num_layers, classes, use_thumbnail):
    block_type, layers, filters = resnet_spec[num_layers]
    resnet = resnet_net_versions[version]
    block = resnet_block_versions[version][block_type]
    return resnet(block, classes, layers, filters, use_thumbnail)

dataset_classes = {'mnist': 10, 'cifar10': 10, 'imagenet': 1000, 'dummy': 1000}

batch_size, dataset, classes = opt.batch_size, opt.dataset, dataset_classes[opt.dataset]

gpus, version = opt.gpus, opt.resnet_version-1

if opt.benchmark:
    batch_size = 32
    dataset = 'dummy'
    classes = 1000
    version = 0


net = get_resnet(version, opt.resnet_layers, classes, opt.thumbnail)

batch_size *= max(1, gpus)

# get dataset iterators
if dataset == 'mnist':
    train_data, val_data = mnist_iterator(batch_size, (1, 32, 32))
elif dataset == 'cifar10':
    train_data, val_data = cifar10_iterator(batch_size, (3, 32, 32))
elif dataset == 'dummy':
    train_data, val_data = dummy_iterator(batch_size, (3, 224, 224))

def test(ctx):
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        metric.update(label, outputs)
    return metric.get()


def train(epoch, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epoch):
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if opt.log_interval:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f'%(
                        epoch, i, batch_size/(time.time()-btic), name, acc))
            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] training: %s=%f'%(epoch, name, acc))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))

        name, val_acc = test(ctx)
        print('[Epoch %d] validation: %s=%f'%(epoch, name, val_acc))

    net.collect_params().save('resnet.params')


if __name__ == '__main__':
    if opt.symbolic:
        data = mx.sym.var('data')
        out = net(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=[mx.gpu(i) for i in range(gpus)] if gpus > 0 else [mx.cpu()])
        mod.fit(train_data, num_epoch=opt.epochs, batch_end_callback = mx.callback.Speedometer(batch_size, 1))
    else:
        net.hybridize()
        train(opt.epochs, [mx.gpu(i) for i in range(gpus)] if gpus > 0 else [mx.cpu()])
