import skimage.feature
import numpy as np
import cv2

def features(img):
    return np.concatenate([hist(img), bin(img), hog(img)])

ORIENTATIONS = 9
PIX_IN_CELL = 8
CELL_IN_BLOCK = 3

def hstack_channels(method, img):
    return np.hstack(map(method,
                         (img[:,:,c]
                          for c in range(img.shape[-1]))))

def chog(channel, 
         orientations=ORIENTATIONS,
         cell=(PIX_IN_CELL, PIX_IN_CELL), 
         block=(CELL_IN_BLOCK, CELL_IN_BLOCK),
         flatten=True):
    return skimage.feature.hog(channel, 
                               orientations=orientations, 
                               pixels_per_cell=cell, 
                               cells_per_block=block, 
                               block_norm='L2-Hys',
                               transform_sqrt=False,
                               visualise=False,
                               feature_vector=flatten)

def hog(img):
    return hstack_channels(chog, img)

BINS = 32

def chist(channel, bins=BINS):
    return np.histogram(channel, bins=bins, range=(0, 1))[0]

def hist(img):
    return hstack_channels(chist, img)

SAMPLE = 32

def bin(img):
    return cv2.resize(img, (SAMPLE, SAMPLE),
                      interpolation=cv2.INTER_NEAREST).ravel()


if __name__ == '__main__':
    from timeit import timeit
    import test_images

    setup = ('import cv2;'
             'import test_images;'
             'import features;'
             'img=cv2.imread(test_images.NAMES[0])[0:64,0:64]')
    
    print('Hog:', hog(cv2.imread(test_images.NAMES[0])[0:64,0:64]).shape)
    print('Hog time: {:.3f}s'.format(timeit('features.hog(img)', setup, number=100)))

    print('Hist:', hist(cv2.imread(test_images.NAMES[0])[0:64,0:64]).shape)
    print('Hist time: {:.3f}s'.format(timeit('features.hist(img)', setup, number=100)))

    print('Bin:', bin(cv2.imread(test_images.NAMES[0])[0:64,0:64]).shape)
    print('Bin time: {:.3f}s'.format(timeit('features.bin(img)', setup, number=100)))

    print('Features:', features(cv2.imread(test_images.NAMES[0])[0:64,0:64]).shape)
    print('Features time: {:.3f}s'.format(timeit('features.features(img)', setup, number=100)))
