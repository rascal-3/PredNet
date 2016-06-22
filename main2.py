# _*_ coding: utf-8 _*_

import argparse
import cv2
import os
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error
import net


class Main(object):

    def __init__(self):

        parser = argparse.ArgumentParser(description='PredNet')

        parser.add_argument('train', help='Input movie filename')
        parser.add_argument('--gpu', '-g', default=0, type=int,
                            help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--root', '-r', default='.',
                            help='Root directory path of image files')
        parser.add_argument('--filename', '-f', default='.',
                            help='Path of input movie file')
        parser.add_argument('--initmodel', default='',
                            help='Initialize the model from given file')
        parser.add_argument('--resume', default='',
                            help='Resume the optimization from snapshot')
        parser.add_argument('--test', dest='test', action='store_true')
        parser.set_defaults(test=False)
        
        self.args = parser.parse_args()

        if self.args.gpu >= 0:
            cuda.check_cuda_available()
        self.xp = cuda.cupy if self.args.gpu >= 0 else np

        # Create Model
        self.in_width = 64
        self.in_height = 64
        self.in_channel = 3

        self.prednet = net.PredNet(self.in_width, self.in_height, (self.in_channel, 48, 96, 192),
                                   (self.in_channel, 48, 96, 192))

        self.model = L.Classifier(self.prednet, lossfun=mean_squared_error)
        self.model.compute_accuracy = False
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        if self.args.gpu >= 0:
            cuda.get_device(self.args.gpu).use()
            self.model.to_gpu()

        # Init/Resume
        if self.args.initmodel:
            print('Load model from', self.args.initmodel)
            serializers.load_npz(self.args.initmodel, self.model)
        if self.args.resume:
            print('Load self.optimizer state from', self.args.resume)
            serializers.load_npz(self.args.resume, self.optimizer)

    def main(self):

        image_list = self.load_image_list(self.args.filename)

        if self.args.test:
            self.prednet.reset_state()
            self.model.zerograds()
            loss = 0
            batch_size = 1
            x_batch = np.ndarray((batch_size, self.in_channel, self.in_height, self.in_width), dtype=np.float32)
            y_batch = np.ndarray((batch_size, self.in_channel, self.in_height, self.in_width), dtype=np.float32)
            for i in range(0, len(image_list)):
                print('frameNo:' + str(i))
                x_batch[0] = self.read_image(image_list[i])
                loss += self.model(chainer.Variable(self.xp.asarray(x_batch)),
                              chainer.Variable(self.xp.asarray(y_batch)))
                loss.unchain_backward()
                loss = 0
                self.model.to_cpu()
                self.write_image(x_batch[0].copy(), 'out/' + str(i) + 'x.jpg')
                self.write_image(self.model.y.data[0].copy(), 'out/' + str(i) + 'y.jpg')
                self.model.to_gpu()

            self.model.to_cpu()
            x_batch[0] = self.model.y.data[0].copy()
            self.model.to_gpu()
            for i in range(len(image_list), len(image_list) + 100):
                print('extended frameNo:' + str(i))
                loss += self.model(chainer.Variable(self.xp.asarray(x_batch)),
                              chainer.Variable(self.xp.asarray(y_batch)))
                loss.unchain_backward()
                loss = 0
                self.model.to_cpu()
                self.write_image(self.model.y.data[0].copy(), 'out/' + str(i) + 'y.jpg')
                x_batch[0] = self.model.y.data[0].copy()
                self.model.to_gpu()

        else:
            for num in range(0, 10000):
                bprop_len = 20
                self.prednet.reset_state()
                self.model.zerograds()
                loss = 0

                batch_size = 1
                x_batch = np.ndarray((batch_size, self.in_channel, self.in_height, self.in_width), dtype=np.float32)
                y_batch = np.ndarray((batch_size, self.in_channel, self.in_height, self.in_width), dtype=np.float32)
                x_batch[0] = self.read_image(image_list[0])

                for i in range(1, len(image_list)):
                    y_batch[0] = self.read_image(image_list[i])
                    loss += self.model(chainer.Variable(self.xp.asarray(x_batch)),
                                  chainer.Variable(self.xp.asarray(y_batch)))

                    print('frameNo:' + str(i))
                    if (i + 1) % bprop_len == 0:
                        self.model.zerograds()
                        loss.backward()
                        loss.unchain_backward()
                        loss = 0
                        self.optimizer.update()
                        self.model.to_cpu()
                        # self.write_image(x_batch[0].copy(), 'out/' + str(num) + '_' + str(i) + 'a.jpg')
                        self.write_image(x_batch[0].copy(), 'out/{}_{}a.jpg'.format(num, i))
                        # self.write_image(self.model.y.data[0].copy(), 'out/' + str(num) + '_' + str(i) + 'b.jpg')
                        self.write_image(self.model.y.data[0].copy(), 'out/{}_{}b.jpg'.format(num, i))
                        # self.write_image(y_batch[0].copy(), 'out/' + str(num) + '_' + str(i) + 'c.jpg')
                        self.write_image(y_batch[0].copy(), 'out/{}_{}c.jpg'.format(num, i))
                        self.model.to_gpu()
                        print('loss:' + str(float(self.model.loss.data)))

                    if i == 1 and (num % 10) == 0:
                        print('save the model')
                        serializers.save_npz('out/' + str(num) + '.model', self.model)
                        print('save the self.optimizer')
                        serializers.save_npz('out/' + str(num) + '.state', self.optimizer)

                    x_batch[0] = y_batch[0]

    def load_image_list(self, filename):
        mov = cv2.VideoCapture(filename)
        images = []
        for i in range(0, int(mov.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))):
            _, img = mov.read()
            self.in_height, self.in_width, self.in_channel = img.shape
            images.append(img)
            cv2.imwrite('{}_{}.jpg'.format(filename, i), img)
        return images

    def read_image(self, path):
        # image = np.asarray(Image.open(path)).transpose(2, 0, 1)
        image = np.asarray(Image.fromarray(path)).transpose(2, 0, 1)
        # print(str(image.shape[0])+'x'+str(image.shape[1])+'x'+str(image.shape[2]))
        top = (image.shape[1] - self.in_height) / 2
        left = (image.shape[2] - self.in_width) / 2
        bottom = self.in_height + top
        right = self.in_width + left
        image = image[:, top:bottom, left:right].astype(np.float32)
        image /= 255
        return image

    @staticmethod
    def write_image(image, path):
        image *= 255
        image = image.transpose(1, 2, 0)
        image = image.astype(np.uint8)
        result = Image.fromarray(image)
        result.save(path)


if __name__ == '__main__':

    main = Main()
    main.main()
