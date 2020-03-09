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

def loop_folder(folder, label, size=(64,64), ftype='.jpg'):
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
    if len(sys.argv) < 4:
        print('python to_gray_csv.py dog_folder cat_folder output_csv')
        sys.exit(0)
    
    dogs = argv[1]
    cats = argv[2]
    output = argv[3]
    
    valid, total, values1 = loop_folder(dogs, 1)
    print()
    print(f'''Converted {valid} / {total} Pictures in {dogs}''')
    print('values1', values1.shape, values1.dtype)
    print()
    
    valid, total, values2 = loop_folder(cats, 0)
    print()
    print(f'''Converted {valid} / {total} Pictures in {cats}''')
    print('values2', values2.shape, values2.dtype)
    print()
    
    values = np.vstack((values1,values2))
    print('Saving as .csv file')
    print('values:', values.shape, values.dtype)
    np.savetxt(output, values, delimiter=',', fmt='%d')
    print()
    
    binary = 'binary.npy'
    if (len(argv) < 5):
        binary = argv[4]
    print('Saving as .npy file (numpy binary matrix)')
    np.save(binary, values)
    print()

if __name__ == '__main__':
    main(sys.argv)
    print('Done')
    