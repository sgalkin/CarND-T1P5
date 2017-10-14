from window import SIZE

class ROI:
    def __init__(self, size):
        self.TOP = 5*size[1]//10
        self.BOTTOM = 9*size[1]//10
        
        self._HORIZON = 1*size[1]//10 # after TOP

        self._osize = size
        self._rsize = [size[0], self.BOTTOM - self.TOP]
        

    def bbox(self, window):
        MH = max(SIZE, min(self._rsize[1], 3*window))
        MW = max(SIZE, min(self._rsize[0], 11*window))

        horizon = (MH*self._HORIZON)//self._rsize[1]
        top = self._HORIZON - horizon 
        bottom = top + MH
    
        left = (self._rsize[0] - MW)//2
        right = left + MW
    
        return (left, self.TOP+top), (right, self.TOP+bottom)


if __name__ == '__main__':
    import moviepy.editor as mpy
    video = mpy.VideoFileClip('test_video.mp4')
    r = ROI(video.size)
    print(r.TOP, r.BOTTOM)
    for w in [256, 128, 64, 32]:
        print(r.bbox(w))
