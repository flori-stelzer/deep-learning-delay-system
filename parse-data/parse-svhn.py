 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data_train = loadmat("train_32x32.mat")
img_train = data_train["X"]  # numpy.array with shape (32, 32, 3, 73257)
labels_train = data_train["y"]  # numpy.array with shape (73257, 1)

data_test = loadmat("test_32x32.mat")
img_test = data_test["X"]  # numpy.array with shape (32, 32, 3, 26032)
labels_test = data_test["y"]  # numpy.array with shape (26032, 1)

# shuffle indices for 6 random batches of size 12209 (i.e. 3 training examples will be dropped)
random_indices = np.random.choice(73257, 73254,replace=False)

def make_files(img_data, labels_data, img_filename, labels_filename):
    # img_data: numpy integer 4D array of shape (32, 32, 3, number of images)
    # labels_data: numpy integer 2D array of shape (number of images, 1)
    number_of_images = img_data.shape[3]
    with open(img_filename, "w") as img_file:
        for line_index in range(number_of_images):
            img_ints = img_data[:,:,:,line_index].flatten()
            line = ""
            for value in img_ints:
                hex_code = hex(value)[2:]
                if len(hex_code) == 1:
                    hex_code = "0" + hex_code
                line += hex_code
            line += "\n"
            img_file.write(line)
            if ((line_index+1) % 5000 == 0):
                print(line_index+1, "of", number_of_images)
    with open(labels_filename, "w") as labels_file:
        line = ""
        for number in labels_data.flatten():
            if number == 10:
                line += "0"
            else:
                line += str(number)
        line += "\n"
        labels_file.write(line)
    return


# save all training data
make_files(img_train, labels_train, "train_images.txt", "train_labels.txt")
print("all-training-data files done")

# save training data split into batches
for i in range(1,7):
    img_filename = "train_images_" + str(i) + ".txt"
    labels_filename = "train_labels_" + str(i) + ".txt"
    batch_indices = random_indices[(i-1)*12209:i*12209]
    make_files(img_train[:,:,:,batch_indices], labels_train[batch_indices,:], img_filename, labels_filename)
    print("training data batch", i, "files done")

# save test data
make_files(img_test, labels_test, "test_images.txt", "test_labels.txt")
print("test data files done")
