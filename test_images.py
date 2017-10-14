import glob
import os

PATH = 'test_images'
NAMES = glob.glob(os.path.join(PATH, 'test*.*'))

if __name__ == '__main__':
    for _ in range(2):
        for n in NAMES:
            print(n)
