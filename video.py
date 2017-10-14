#!/usr/bin/env python

import argparse
import moviepy.editor as mpy

import cv2

from model import Model
from roi import ROI
from tracker import Tracker
from detector import detect

WINDOWS = [152, 128, 72, 56]
THRESHOLD = 3

def recode(video, model):
    roi = ROI(video.size)
    tracker = Tracker()

    def track(frame):
        bb, _ = detect(frame, WINDOWS, THRESHOLD, model, roi)
        tracker.track(bb)
        for b in tracker.objects():
            cv2.rectangle(frame, (int(b[1]), int(b[0])), (int(b[3]), int(b[2])), 
                          color=(255, 0, 0), thickness=3)
        return frame

    return video.fl_image(track)

def main(args):
    m = Model.load(args.model)
    video = mpy.VideoFileClip(args.video)
    video = video.subclip(args.b, args.e)
    video = recode(video, m)
    video.write_videofile(args.filename, audio=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video converseion utility')
    parser.add_argument('video', type=str,
                        help='Input video file')
    parser.add_argument('-o', dest='filename', type=str, required=True,
                        help='Output video file')
    parser.add_argument('-m', dest='model', type=str, required=True,
                        help='Model to use')
    parser.add_argument('-b', type=int, default=0, help='Start time')
    parser.add_argument('-e', type=int, default=-1, help='End time')
    main(parser.parse_args())

