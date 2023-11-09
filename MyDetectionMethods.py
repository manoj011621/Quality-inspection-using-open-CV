import numpy as np
import cv2 as cv
import matplotlib as plt

class MyDetectionMethod:
    def __init__(self,image):
        self.image=image
    def gray_scale(self):
        #convert to gray scale
        imgray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        return imgray
    def remove_noise(self,imgray):
        #removed noise
        blur_img=cv.GaussianBlur(imgray, (7, 7), 0)
        return blur_img
    def get_threshold(self,blur_img):
        #adaptive threshold
        threshold_img=cv.adaptiveThreshold(blur_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 19, 5)
        return threshold_img
    def find_contours(self,threshold_img):
        #finding contours
        contours, hierarchy = cv.findContours(threshold_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_in_frame = []

    # filter smaller objects less than 2000 from contours
        for contour in contours:
            contour_area = cv.contourArea(contour)
            if contour_area > 2000:
                contours_in_frame.append(contour)

        return contours_in_frame

        
    def draw_contours(self,contours):
         # create an empty image for contours
        img_contours = np.zeros(self.image.shape)
        # draw the contours on the empty image
        cv.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
        return img_contours


