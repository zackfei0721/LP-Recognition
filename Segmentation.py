import cv2
import cv2_utils
import numpy as np
from PIL import Image
import os.path
from skimage import io, data
import matplotlib.pyplot as plt


def insider(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x1 > x2 and y1 > y2 and x1+w1<x2+w2 and y1+h1<y2+h2:
        return True
    return False


def myResize(src, w, h):
    src_h = src.shape[0]
    src_w = src.shape[1]
    width = int(src_w * h / src_h)
    src = cv2.resize(src, (width, h))
    if width > w:
        if (width - w) % 2 == 0:
            dst = src[0:h, int((width - w)/2):int((width - w)/2) + w]
        else:
            m_dst = src[0:h, 0:width-1]
            dst = m_dst[0:h, int((width - 1 - w)):int((width -1 - w)/2) + w]
    elif width < w:
        if (w - width) % 2 == 0:
            dst = cv2.copyMakeBorder(src, top=0, bottom=0, left=int((w - width) / 2), right=int((w - width) / 2), borderType=cv2.BORDER_CONSTANT, value=[0])
        else:
            m_dst = src[0:h, 0:width - 1]
            dst = cv2.copyMakeBorder(m_dst, top=0, bottom=0, left=int((w - width + 1) / 2), right=int((w - width + 1) / 2), borderType=cv2.BORDER_CONSTANT, value=[0])
    else:
        dst = src
    return dst


def Segmentation(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thres_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('ROI_thres_img', thres_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_img = cv2.dilate(thres_img, kernel, iterations=1)

    img, contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    word_rects = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if w > img.shape[1] * 0.5 or h < img.shape[0] * 0.5:
            continue
        else:
            word_rects.append(rect)

    for r1 in word_rects:
        for r2 in word_rects:
            if insider(r1, r2):
                word_rects.remove(r1)

    word_rects = sorted(word_rects, key=lambda rect: rect[0])

    word_images = []
    for rect in word_rects:
        x, y, w, h = rect
        img = myResize(thres_img[y:y + h, x:x + w], 32, 40)
        img = np.array(img, dtype=np.float)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        word_images.append(img)

    plt.figure(1)
    for i, roi in enumerate(word_images):
        plt.subplot(171 + i)
        plt.imshow(roi, cmap=plt.cm.gray)
        plt.title('%dth' % (i + 1))
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('test/test(7)_crop.jpg')
    Segmentation(img)
