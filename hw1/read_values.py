import struct
import numpy as np
NUM_IMAGES = 60000
NUM_PIXELS = 784
HEADER_DATA = 16
HEADER_LABEL = 8 

data = open('data', 'rb')
labels = open('labels', 'rb')
data.read(HEADER_DATA)
labels.read(HEADER_LABEL)

counts = [0] * 10
values1 = [0] * 10
values2 = [0] * 10
for img in xrange(NUM_IMAGES):
    label = struct.unpack('B', labels.read(1))[0]
    img = struct.unpack('784B', data.read(NUM_PIXELS))
    
    counts[label] += 1
    img2 = np.reshape(np.array(img, dtype=float), (28,28))
    values1[label] += img2[6,6]
    values2[label] += img2[13,13]

values1[:] = [values1[i] / counts[i] for i in xrange(10)]
values2[:] = [values2[i] / counts[i] for i in xrange(10)]

print values1
print values2
