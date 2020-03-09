import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
from pathlib import Path

def load(file_path):
    return cv.imread(file_path)

def gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def scale(image, size):
    return cv.resize(image, size, interpolation=cv.INTER_AREA)

def show(image, gray, scale):
    cv.imshow('image', image)
    cv.imshow('gray', gray)
    cv.imshow('scaled', scale)

def loop_folder(folder, size=(64,64), label = 1, ftype='.jpg'):
    failed = 0
    images = 0
    values = []
    for p in Path(folder).glob('**/*' + ftype):
        try:
            i = load(str(p))
            g = gray(i)
            s = scale(g, size)
            # show(i, g, s)
        except Exception as e:
            failed += 1
            sys.stderr.write(f'''{failed} - {p} - {e}''')
        else:
            v = np.append(s.ravel(), label)
            values.append(v)
            images += 1
    return (images, failed + images, np.vstack(values))

def main(argv):
    if len(sys.argv) < 6:
        print('python to_gray_csv.py width height dog_folder cat_folder output_csv')
        sys.exit(0)
    
    size = (int(argv[1]), int(argv[2]))
    dogs = argv[3]
    cats = argv[4]
    output = argv[5]
    
    valid, total, values1 = loop_folder(dogs, size, 1)
    print()
    print(f'''Converted {valid} / {total} Pictures in {dogs}''')
    print('values1', values1.shape, values1.dtype)
    print()
    
    valid, total, values2 = loop_folder(cats, size)
    print()
    print(f'''Converted {valid} / {total} Pictures in {cats}''')
    print('values2', values2.shape, values2.dtype)
    print()
    
    values = np.vstack((values1,values2))
    print('values:', values.shape, values.dtype)
    np.savetxt(output, values, delimiter=',')
    print()

if __name__ == '__main__':
    main(sys.argv)
    print('Done')
    