import functools
import cv2
import numpy as np

from skimage.measure import label

from window import SIZE
from window import fast_sliding_window
from image import convert

AREA_THRESHOLD = 0.75 * SIZE**2

def scale_roi(frame, w, tl, br):
    subframe = frame[tl[1]:br[1], tl[0]:br[0]]
    size = (max(SIZE, (d*SIZE)//w) for d in subframe.shape[-2::-1])
    return cv2.resize(subframe, tuple(size), interpolation=cv2.INTER_LINEAR)

def restore_position(position, w, tl, br):
    return tuple(tl[i] + (d*w)//SIZE
                 for i, d in enumerate(position))

def scan(frame, windows, model, roi):
    def _scan(frame, window, model, roi):
        lt, rb = roi.bbox(window)
        #cv2.rectangle(frame, lt, rb, color=(255,255,255), thickness=4)
        subimage = scale_roi(frame, window, lt, rb)
        cvt = convert(subimage)
        position, features = zip(*fast_sliding_window(cvt, window//(2*SIZE)+1))

        prediction = model.predict(features)
        return ((*restore_position(xy, window, lt, rb), window) 
                for xy in np.array(position)[np.where(prediction == 1)])
    
    return (xyw for bboxes in map(lambda w: _scan(frame, w, model, roi),
                                  windows)
                for xyw in bboxes)


def heatmap(frame, bboxes):
    m = np.zeros(frame.shape[:2], dtype=np.float32)
    for (x, y, w) in bboxes:
        m[y:y+w, x:x+w] += 1
    return m
    

def threshold(heat, th):
    h = heat > th
    l, n = label(h, return_num=True)
    for i in range(1, n + 1):
        nz = (l == i).nonzero()
        bbox = *np.min(nz, axis=1), *np.max(nz, axis=1)
        if (bbox[2] - bbox[0])*(bbox[3] - bbox[1]) > AREA_THRESHOLD:
            yield bbox

            
def detect(frame, windows, th, model, roi):
    candidates = scan(frame, windows, model, roi)
    heat = heatmap(frame, candidates)
    bb = threshold(heat, th)
    return bb, heat


if __name__ == '__main__':
    import numpy as np
    import moviepy.editor as mpy

    from roi import ROI
    from model import Model
    import pipeline
    
    video = mpy.VideoFileClip('test_video.mp4')
    r = ROI(video.size)
    pf, nf, pl, nl = pipeline.read('test_sets/vehicles', 'test_sets/non-vehicles')
    m = Model()
    m.train([1], np.vstack((pf,nf)), np.hstack((pl,nl)))

    bb, _ = detect(video.get_frame(0), [128], 4, m, r)
    print(list(bb))
