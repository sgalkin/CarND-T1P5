from trackable import Trackable

class Tracker:
    def __init__(self):
        self._objects = []
        
    def track(self, bboxes):
        for o in self._objects:
            o.unseen()
    
        for bb in bboxes: 
            found = False
            for o in filter(lambda o: not found and o.match(bb), self._objects):
                o.push(bb)
                found = True
            if not found:
                self._objects.append(Trackable(bb))

        self._objects = list(filter(lambda o: not o.delete(), self._objects))

    def objects(self):
        return (o.bbox() for o in filter(lambda o: o.good(), self._objects))

if __name__ == '__main__':
    import trackable

    t = Tracker()
    bb1 = [1,2,3,4]
    t.track(bb1)
    assert(len(t.objects()) == 0)
    for _ in range(trackable.SEEN_THRESHOLD):
        t.track(bb1)
    assert(len(t.objects()) == 1)
    assert(all(next(t.objects()) == bb1))

    bb2 = [100, 200, 400, 500]
    t.track(bb2)
    assert(len(t.objects()) == 1)
    for _ in range(max(trackable.SEEN_THRESHOLD, trackable.LOST_THRESHOLD) + 1):
        t.track(bb2)
    assert(len(t.objects()) == 1)
    assert(all(next(t.objects()) == bb2))
