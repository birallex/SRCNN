import os
import cv2
import h5py
import numpy as np

BLOCK_STEP = 16
BLOCK_SIZE = 32
scale = 2
Patch_size = 32
label_size = 20
conv_side = 6
Random_Crop = 30


def prepare_valid_data(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = np.zeros((nums * Random_Crop, 1, Patch_size, Patch_size), dtype=np.double)
    label = np.zeros((nums * Random_Crop, 1, label_size, label_size), dtype=np.double)

    for i in range(nums):
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        shape = hr_img.shape

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        lr_img = cv2.resize(hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))
        Points_x = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
        Points_y = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
        for j in range(Random_Crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * Random_Crop + j, 0, :, :] = lr_patch
            label[i * Random_Crop + j, 0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]

    data = np.transpose(data, (0, 2, 3, 1))
    label = np.transpose(label, (0, 2, 3, 1))
    return data, label


def prepare_train_data(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()
    data = []
    label = []
    for i in range(nums):
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        shape = hr_img.shape
        lr_img = cv2.resize(hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        width_num = int((shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP)
        height_num = int((shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP)
        for k in range(width_num):
            for j in range(height_num):
                x = k * BLOCK_STEP
                y = j * BLOCK_STEP
                hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                lr = np.zeros((1, Patch_size, Patch_size), dtype=np.double)
                hr = np.zeros((1, label_size, label_size), dtype=np.double)

                lr[0, :, :] = lr_patch
                hr[0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
                data.append(lr)
                label.append(hr)

    data = np.array(data, dtype=float)
    label = np.array(label, dtype=float)
    data = np.transpose(data, (0, 2, 3, 1))
    label = np.transpose(label, (0, 2, 3, 1))
    return data, label


def write_hdf5(data, labels, output_filename):
    x = data.astype(np.float32)
    y = labels.astype(np.float32)
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)

    
if __name__ == "__main__":
    data, label = prepare_train_data("photos/")
    write_hdf5(data, label, "train.h5")
    data, label = prepare_valid_data("valid_photos/")
    write_hdf5(data, label, "test.h5")