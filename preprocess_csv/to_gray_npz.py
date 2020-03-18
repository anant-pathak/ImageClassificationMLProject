import numpy as np
import cv2 as cv
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

def cvt_img(file, size=(64,64)):
    image = cv.imread(file)
    gray  = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    small = cv.resize(gray, size, interpolation=cv.INTER_AREA)
    return small

def cvt_dir(dir, size=(64,64), ftype='.jpg'):
    failed = 0
    success = 0
    images = []
    print(f'''Converting "{ftype}" in directory "{dir}""''')
    for p in Path(dir).glob('**/*' + ftype):
        try:
            img = cvt_img(str(p))
        except Exception as e:
            failed += 1
            sys.stderr.write(f'''{str(p)} => {e}''')
        else:
            success += 1
            images.append(img.ravel())
    print(f'''Converted {success}/{success + failed} "{ftype}" images''')
    return np.vstack(images)

def main(argv):
    if (len(argv) < 3):
        print('python to_gray_npz dog_dir cat_dir output_file')
        sys.exit()
    
    dogs = cvt_dir(argv[1])
    # np.random.shuffle(dogs)
    print('dogs:', dogs.shape, dogs.dtype)

    cats = cvt_dir(argv[2])
    # np.random.shuffle(cats)
    print('cats:', cats.shape, cats.dtype)

    small_dim = min(dogs.shape[0], cats.shape[0])
    array_type = dogs.dtype
    assert(array_type == cats.dtype)
    print('small dimension:', small_dim, 'data type:', array_type)

    data = np.vstack((dogs[:small_dim,:], cats[:small_dim,:]))
    print('data:', data.shape, data.dtype)

    labels = np.vstack((np.ones((small_dim,1), dtype=array_type), np.zeros((small_dim,1), dtype=array_type)))
    print('labels:', labels.shape, labels.dtype)

    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.1)
    print('data train:', data_train.shape, data_train.dtype)
    print('label train:', label_train.shape, label_train.dtype)
    print('data test:', data_test.shape, data_test.dtype)
    print('label test:', label_test.shape, label_test.dtype)

    print(f'''Outputing to "{Path(argv[3])}"''')
    np.savez_compressed(argv[3], data_train=data_train, label_train=label_train, data_test=data_test, label_test=label_test)

if __name__ == '__main__':
    main(sys.argv)