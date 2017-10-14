#!/usr/bin/env python

import argparse
import logging

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import pipeline
from model import Model

TEST_SIZE = 0.2

def main(args):
    pf, nf, pl, nl = pipeline.read(args.p, args.n)
    logging.info('Found %d positive examples', len(pf))
    logging.info('Found %d negative examples', len(nf))

    f, l = shuffle(np.vstack((pf, nf)), np.hstack((pl, nl)))
    ft, fv, lt, lv = train_test_split(f, l, test_size=args.s)
    
    logging.info('Train size: %d examples', len(ft))
    logging.info('Test size: %d examples', len(fv))
    
    logging.debug('Training model')
    m = Model()
    m.train(args.C, ft, lt)
    logging.debug('Training completed')

    logging.debug('Evaluating model')
    pt, pv = (m.predict(x) for x in (ft, fv))
    logging.info('Train set accuracy/f1-score: %.5f/%.5f',
                 accuracy_score(lt, pt), f1_score(lt, pt))
    logging.info('Test set accuracy/f1-scoere: %.5f/%.5f',
                 accuracy_score(lv, pv), f1_score(lv, pv))
    m.store(args.model)
    logging.info('Model saved')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('model', type=str,
                        help='Path to model file')
    parser.add_argument('-p', type=str, required=True,
                        help='Path to positive samples')
    parser.add_argument('-n', type=str, required=True,
                        help='Path to negavtive sameples')
    parser.add_argument('-C', nargs='+', default=[1], type=int,
                        help='List of C values')
    parser.add_argument('-s', type=float, default=TEST_SIZE,
                        help='Test set size')

    main(parser.parse_args())
    
