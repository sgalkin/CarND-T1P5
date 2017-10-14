import math
from averager import Averager
from window import SIZE

AVG_SERIES = 12
SEEN_THRESHOLD = 4
LOST_THRESHOLD = 8

DIST_THRESHOLD = SIZE * 1

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)
    
def centroid(bbox):
    return (bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2

class Trackable:
    def __init__(self, bbox):
        self._averager = Averager(AVG_SERIES, len(bbox))
        self._seen = 0
        self._unseen = 0
        self.push(bbox)
        
    def match(self, bbox):
        return distance(centroid(bbox),
                        centroid(self._averager.latest())) < DIST_THRESHOLD
    
    def bbox(self):
        return self._averager.mean()
        
    def push(self, bbox):
        self._averager.push(bbox)
        self._seen += 1
        self._unseen = max(0, self._unseen - 1)
        
    def unseen(self):
        self._unseen += 1
        if self.delete():
            self._seen = 0
            
    def good(self):
        return self._seen > SEEN_THRESHOLD
    
    def delete(self):
        return self._unseen > LOST_THRESHOLD

if __name__ == '__main__':
    bb = [1,2,3,4]
    t = Trackable(bb)
    assert(all(t.bbox() == bb))

    for _ in range(SEEN_THRESHOLD):
        assert(not t.good())
        assert(not t.delete())
        t.push(bb)
    assert(t.good())
    assert(all(t.bbox() == bb))
    
    for _ in range(LOST_THRESHOLD):
        t.unseen()
        assert(t.good())
        assert(not t.delete())
    t.unseen()
    assert(t.delete())
    assert(not t.good())
