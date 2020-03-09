# Preprocess Images Into .csv file
Converts all the images in specified folders into a .csv file that
contains pixels values with label as last element in row.

A label of 1 means it is a dog, a label of 0 means it is a cat.

## Running
```
python to_gray_csv.py dog_image_folder cat_image_folder csv_file_name
```
## Notes
* This may take a little bit to run... so be patient.
* OpenCV sometimes has trouble opening an image, therefore this program will skip any image that causes an issue. ___See printed output for how many images the program converted.___
* The final .csv file can be quite large.

## Setup (with requirements.txt)
pip install -r /path/requirements.txt

## Setup (manual)
pip install numpy
pip install matplotlib
pip install opencv-python