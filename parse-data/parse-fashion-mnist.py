"""
script to convert fashion-MNIST data to more practical file type (txt files)

images contain 28*28 = 784 pixels

new files: image files are txt files
           each line represents one image
           pixels are saved as 2-digit hex-numbers between 00 and ff (255)
           00 is white, ff is black (reverse to normal image file, but in correspondence to MNIST dataset)
           i.e. a file containing 10000 images will have 10000 lines of length 784 * 2 = 1568
           the end of each line is marked by the EOF symbol '\n'
           
           label files are also txt files
           labels are stored in one line (one long string) with characters 0, ..., 9.
           i.e. the label file for an image file with 10000 images contains one line of length 10000
           the end of the line is marked by the EOF symbol '\n'
           
The MNIST dataset contains 60000 training data points, but not 6000 of each class, instead:
 0: 5923,
 1: 6742,
 2: 5958,
 3: 6131,
 4: 5842,
 5: 5421,
 6: 5918,
 7: 6265,
 8: 5851,
 9: 5949.
 
It contains 10000 test data points:
 0: 5923,
 1: 6742,
 2: 5958,
 3: 6131,
 4: 5842,
 5: 5421,
 6: 5918,
 7: 6265,
 8: 5851,
 9: 5949.
 
The training dataset will be randomly splitted into 6 batches of 10000 data points each for cross-validation.
The files are called train_images_n.txt and train_labels_n.txt, n = 1, ..., 6.
The test dataset will be saved in the files test_images.txt and test_labels.txt

Since the data is already mixed in the MNIST dataset, we can just split the training dataset into the batches.
We do not need to mix the data again by ourselves.
           

"""

# for the plot
import matplotlib
matplotlib.rcParams["backend"] = "qt5Agg"
# uncomment the above line if you run
# the example script on a TU-Berlin computer
import matplotlib.pyplot as plt

import numpy as np
from mnist import MNIST




mndata = MNIST('./orig_data')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()


# how many of each numbers in dataset?
train_dict = dict()
for label in labels_train:
    if label in train_dict:
        train_dict[label] += 1
    else:
        train_dict[label] = 1

test_dict = dict()
for label in labels_test:
    if label in test_dict:
        test_dict[label] += 1
    else:
        test_dict[label] = 1

        
# total number:
total_train = 0
for label in train_dict:
    total_train += train_dict[label]
print("total number of training data points:", total_train)

for key in sorted(train_dict):
    print("count of digit", key, "in training dataset:", train_dict[key]) 

total_test = 0
for label in test_dict:
    total_test += test_dict[label]
    
print("---   ---   ---")
print("total number of test data points:", total_test)

for key in sorted(test_dict):
    print("count of digit", key, "in test dataset:", test_dict[key]) 
    
# save test data in txt files
labels_test_string = ""
for label in labels_test:
    labels_test_string += str(label)
with open("test_labels.txt", "w") as textfile:
    textfile.write(labels_test_string + "\n")

first_image = True
for image in images_test:
    image_string = ""
    for pixel in image:
        hex_literal = hex(pixel)
        if pixel < 16:
            hex_code = "0" + hex_literal[2]
        else:
            hex_code = hex_literal[2:]
        image_string += hex_code
    if first_image:
        with open("test_images.txt", "w") as textfile:
            textfile.write(image_string + "\n")
            first_image = False
    else:
        with open("test_images.txt", "a") as textfile:
            textfile.write(image_string + "\n")

# save training data in txt files
for n in range(6):
    labels_train_string = ""
    batch_dict = dict()  # for counting
    for label in labels_train[10000*n:10000*(n+1)]:
        labels_train_string += str(label)
        # only for counting:
        if label in batch_dict:
            batch_dict[label] += 1
        else:
            batch_dict[label] = 1
    file_name = "train_labels_" + str(n+1) + ".txt"
    with open(file_name, "w") as textfile:
        textfile.write(labels_train_string + "\n")
    print("---   ---   ---")
    for key in sorted(batch_dict):
        print("count of digit", key, "in training batch " + str(n + 1) + ":", batch_dict[key]) 

    first_image = True
    for image in images_train[10000*n:10000*(n+1)]:
        image_string = ""
        for pixel in image:
            hex_literal = hex(pixel)
            if pixel < 16:
                hex_code = "0" + hex_literal[2]
            else:
                hex_code = hex_literal[2:]
            image_string += hex_code
        file_name = "train_images_" + str(n+1) + ".txt"
        if first_image:
            with open(file_name, "w") as textfile:
                textfile.write(image_string + "\n")
                first_image = False
        else:
            with open(file_name, "a") as textfile:
                textfile.write(image_string + "\n")
    


# make 6 training batches with 10000 data points each

