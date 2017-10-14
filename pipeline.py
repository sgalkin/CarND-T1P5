import concurrent.futures

import functools
import itertools
import random
import fnmatch
import os

import numpy as np

from features import features
import image

PIPE = (image.load, image.flip, image.augmentation)

def read(positive, negative):
    with concurrent.futures.ProcessPoolExecutor() as e:
        nf, pf = (scrub(PIPE, dir, e) for dir in (negative, positive))

    nl, pl = np.zeros(len(nf)), np.ones(len(pf))
    assert(len(nf) == len(nl))
    assert(len(pf) == len(pl))
    return pf, nf, pl, nl

def walk(path, filter):
    return (os.path.join(r, n) 
            for r, _, f in os.walk(path) 
            for n in fnmatch.filter(f, filter))

def pipeline(functions, initial):
    return functools.reduce(lambda v, g: (nv
                                          for ov in v
                                          for nv in g(ov)), 
                            functions, 
                            (initial,))

def feature_extractor(args):
    f, i, seed = args
    np.random.seed(seed) # set different random seed for each process in pool
    return np.vstack(features(image.convert(img))
                     for img in pipeline(f, i))

def scrub(functions, path, e):
    return np.vstack(e.map(feature_extractor, 
                           zip(itertools.cycle((functions,)), 
                               walk(path, '*.png'),
                               (random.randint(0, 10**6)
                                for _ in itertools.cycle((0,))),
                           ), 
                           chunksize=500))

if __name__ == '__main__':
    pf, nf, pl, nl = read('test_sets/vehicles', 'test_sets/non-vehicles')
    print('Positive data set:', len(pf), 'features:', len(pf[0]))
    print('Negative data set:', len(nf), 'features:', len(nf[0]))
