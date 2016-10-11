import struct
import numpy.matlib as np
import math
NUM_TRAIN_IMAGES = 60000
NUM_TEST_IMAGES = 10000
NUM_PIXELS = 784
HEADER_DATA = 16
HEADER_LABEL = 8 

train_data = open('../mnist_data/60k_train_images', 'rb')
train_labels = open('../mnist_data/60k_train_labels', 'rb')
train_data.read(HEADER_DATA)
train_labels.read(HEADER_LABEL)

test_data = open('../mnist_data/10k_test_images', 'rb')
test_labels = open('../mnist_data/10k_test_labels', 'rb')
test_data.read(HEADER_DATA)
test_labels.read(HEADER_LABEL)

digits = 10
pixels = 784

class NeuralNet:
    def __init__(self):
        self.lr = 0.5
        self.bias = np.zeros((digits, 1))
        self.weights = np.zeros((digits, pixels))

    def train(self, img, label):
        h1 = self.weights * img + self.bias
        softmax = np.exp(h1)
        softmax /= np.sum(softmax)
        error = -np.log(softmax[label])

        dedh = np.matrix([softmax.item(j) if j != label else softmax.item(j) - 1 for j in range(digits)])

        dedw = dedh.T * img.T

        self.bias -= np.multiply(dedh.T, self.lr)
        self.weights -= np.multiply(dedw, self.lr)

    def classify(self, img):
        h1 = self.weights * img + self.bias
        softmax = np.exp(h1)
        softmax /= np.sum(softmax)
        return np.argmax(softmax)


net = NeuralNet()
# forward step
for _ in range(10000):
    label = struct.unpack('B', train_labels.read(1))[0]
    img = np.divide(np.matrix(struct.unpack('784B', train_data.read(NUM_PIXELS)), dtype=float).T, 255)
    net.train(img, label)

correct = [0] * NUM_TEST_IMAGES

for i in range(NUM_TEST_IMAGES):
    label = struct.unpack('B', test_labels.read(1))[0]
    img = np.divide(np.matrix(struct.unpack('784B', test_data.read(NUM_PIXELS)), dtype=float).T, 255)
    if net.classify(img) == int(label):
        correct[i] = 1
print (float(sum(correct)) / NUM_TEST_IMAGES * 100, "% accuracy")


