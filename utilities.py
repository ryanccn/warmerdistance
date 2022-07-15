import cv2
import numpy as np


def letterbox_image(image, size):
    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scalew = w / iw
    scaleh = h / ih
    nw = int(iw * scalew)
    nh = int(ih * scaleh)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    new_image = np.zeros((size[1], size[0], 3), np.uint8)
    new_image.fill(128)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image[dy : dy + nh, dx : dx + nw, :] = image

    return [new_image, (scalew, scaleh), (nw, nh)]
