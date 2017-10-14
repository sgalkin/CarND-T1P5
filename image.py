import numpy as np
import cv2

def passthrough(image):
    return image

def load(path, transformation=passthrough):
    img = cv2.imread(path)
    timg = transformation(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    yield timg
    
def flip(rgb):
    yield rgb
    yield cv2.flip(rgb, 1)
    
def augmentation(rgb, count=2):
    yield rgb
    if count <= 0:
        return
    
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)

    for i in range(count):
        h = 0 # it's not safe to change color of the objects
        l = np.random.randint(-30, 31)
        s = np.random.randint(-20, 21)
        
        h_a = hls[:,:,0]
        l_a = cv2.add(hls[:,:,1], l)
        s_a = cv2.add(hls[:,:,2], s)
        img = cv2.cvtColor(np.dstack((h_a, l_a, s_a)), cv2.COLOR_HLS2RGB)
        yield img

def convert(rgb):
    targets = [
        #cv2.COLOR_RGB2LAB, 
        #cv2.COLOR_RGB2HLS,
        cv2.COLOR_RGB2YCrCb,
        #cv2.COLOR_RGB2GRAY,
    ]
    rgb = np.sqrt(np.float32(rgb)/255.)
    if len(targets) == 1:
        return cv2.cvtColor(rgb, targets[0])
    return np.dstack(cv2.cvtColor(rgb, dst) for dst in targets)
