import features
import cv2
import numpy as np

SIZE = 64

def fast_sliding_window(img, cells_per_step):
    scale = SIZE//features.SAMPLE
    n_blocks_per_window = (SIZE//features.PIX_IN_CELL) - features.CELL_IN_BLOCK + 1

    def step(args):
        x_cell, y_cell, hog, img = args

        x_pix, y_pix = (d*features.PIX_IN_CELL for d in (x_cell, y_cell))

        subimage = img[y_pix//scale:y_pix//scale+features.SAMPLE, 
                       x_pix//scale:x_pix//scale+features.SAMPLE]

        g = features.hist(subimage)
        s = subimage.ravel()
        h = hog[:, 
                y_cell:y_cell + n_blocks_per_window, 
                x_cell:x_cell + n_blocks_per_window].ravel()
        return (x_pix, y_pix), np.concatenate([g, s, h])

    hog = np.vstack(np.expand_dims(features.chog(img[:,:,c], flatten=False), 0)
                    for c in range(img.shape[-1]))
    
    scaled_img = cv2.resize(img, 
                            (img.shape[1]//scale, img.shape[0]//scale),
                            interpolation=cv2.INTER_NEAREST)
    
    # Define blocks and steps
    # The computations only valid fro square blocks
    n_yblocks, n_xblocks = ((d//features.PIX_IN_CELL) - features.CELL_IN_BLOCK + 1
                            for d in img.shape[:2])
        
    n_ysteps, n_xsteps = ((n_blocks - n_blocks_per_window)//cells_per_step
                          for n_blocks in (n_yblocks, n_xblocks))
    return map(step, 
               ((x*cells_per_step, y*cells_per_step, hog, scaled_img) 
                for x in range(n_xsteps+1) 
                for y in range(n_ysteps+1)))

# step in 0.1 of SIZE
def slow_sliding_window(img, x_step, y_step):
    return (((x, y), features.features(img[y:y+SIZE,x:x+SIZE]))
            for x in range(img.shape[1]-SIZE, -1, -(x_step*SIZE)//10)
            for y in range(img.shape[0]-SIZE, -1, -(y_step*SIZE)//10))

if __name__ == '__main__':
    import numpy as np
    img = np.zeros((SIZE*3, SIZE*2, 3))
    
    assert(len(list(slow_sliding_window(img, 10, 10))) == 6)
    assert(len(list(slow_sliding_window(img, 5, 5))) == 15)

    assert(len(list(fast_sliding_window(img, 8))) == 6)
    
