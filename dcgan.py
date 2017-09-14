# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import chainer
from chainer import training
from chainer.training import extensions
from net import Discriminator
from net import Generator
from updater import DCGANUpdater
from visualize import out_generated_image

def main():
    parser = argparse.ArgumentParser(description='DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result image')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--gennum','-v', default=10,
                        help='visualize image rows and columns number')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_hidden: {}'.format(args.n_hidden))
    print('# epoch: {}'.format(args.epoch))
    print('')

    #学習モデルの作成
    gen = Generator(n_hidden=args.n_hidden)
    dis = Discriminator()

    if args.gpu >= 0:
        #modelをGPU用に変換
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    #oputimizerのsetup
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    if args.dataset == '':
        #データセットの読み込み defaultはcifar10
        train, _ = chainer.datasets.get_cifar10(withlabel=False, scale=255.)
    else:
        all_files = os.listdir(args.dataset)
        image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]
        print('{} contains {} image files'
              .format(args.dataset, len(image_files)))
        train = chainer.datasets\
            .ImageDataset(paths=image_files, root=args.dataset)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    #trainerのセットアップ
    updater = DCGANUpdater(
        models=(gen, dis),
        iterator=train_iter,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.epoch}.npz'),
        trigger=(100,'epoch'))
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.epoch}.npz'), trigger=(100,'epoch'))
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.epoch}.npz'), trigger=(100,'epoch'))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, dis,
            int(args.gennum),int(args.gennum), args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        #学習済みmodelの読み込み
        chainer.serializers.load_npz(args.resume, trainer)

    #学習の実行
    trainer.run()
    chainer.serializers.save_npz("genmodel{0}".format(args.datasets), trainer)

if __name__ == '__main__':
    main()
