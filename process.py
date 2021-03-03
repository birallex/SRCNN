import cv2
import numpy as np
import os

from models.production_model import production_model


def predict(name):
    srcnn_model = production_model()
    srcnn_model.load_weights("weights/new_weights.h5")
    IMG_NAME = "input/" + name
    name, extention = name.split(".")
    OUTPUT_NAME = name + "_srcnn." + extention
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y_img = img[:, :, 0]
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite("output/" + OUTPUT_NAME, img)


if __name__ == "__main__":
    names = os.listdir("input/")
    for name in names:
        predict(name)