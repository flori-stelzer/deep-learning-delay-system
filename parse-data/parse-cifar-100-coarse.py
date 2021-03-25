 
import numpy as np
import matplotlib.pyplot as plt
import pickle


def unpickle(file):    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def make_files(data_batch, img_filename, labels_filename):
    img_array = data_batch[b"data"]
    with open(img_filename, "w") as img_file:
        for line_index in range(10000):
            img_ints = img_array[line_index, :]
            # reshape img_ints to format (r11, g11, b11, r12, g12, b12, ...)
            img_ints_reshaped = np.zeros(3072, dtype=np.int64)
            for row in range(32):
                for col in range(32):
                    r = img_ints[32*row + col]
                    g = img_ints[1024 + 32*row + col]
                    b = img_ints[2048 + 32*row + col]
                    img_ints_reshaped[3*32*row + 3*col] = r
                    img_ints_reshaped[3*32*row + 3*col + 1] = g
                    img_ints_reshaped[3*32*row + 3*col + 2] = b
            line = ""
            for value in img_ints_reshaped:
                hex_code = hex(value)[2:]
                if len(hex_code) == 1:
                    hex_code = "0" + hex_code
                line += hex_code
            line += "\n"
            img_file.write(line)    
    label_list = data_batch[b"coarse_labels"]
    with open(labels_filename, "w") as labels_file:
        line = ""
        for number in label_list:
            if number < 10:
                line += "0" + str(number)
            else:
                line += str(number)
        line += "\n"
        labels_file.write(line)
    return


for i in range(1,6):
    fname = "train"
    img_filename = "train_images_" + str(i) + ".txt"
    labels_filename = "train_labels_" + str(i) + ".txt"
    data_batch = unpickle(fname)
    data_batch[b"data"] = data_batch[b"data"][(i - 1)*10000:i*10000, :]
    data_batch[b"coarse_labels"] = data_batch[b"coarse_labels"][(i - 1)*10000:i*10000]
    make_files(data_batch, img_filename, labels_filename)

fname = "test"
img_filename = "test_images.txt"
labels_filename = "test_labels.txt"
data_batch = unpickle(fname)
make_files(data_batch, img_filename, labels_filename)
