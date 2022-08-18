import cv2
import numpy as np
import matplotlib.pyplot as plt

class WordSegmentation():
    def __init__(self):
        self.image = None
        self.words_list = []

    def resize_image(self, image):
        height, width, _ = image.shape
        if width > 1000:
            new_width = 1000
            factor = width / height
            new_height = int(new_width / factor)

            new_image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
            plt.imshow(new_image)
        return new_image

    def thresholding(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(image_gray, 80, 255, cv2.THRESH_BINARY_INV)
        # plt.imshow(thresholded_image, cmap='gray')
        return thresholded_image

    def dilation(self, image):
        kernel = np.ones((3,85), np.uint8)
        dilated_image = cv2.dilate(image, kernel, iterations = 1)
        # plt.imshow(dilated_image, cmap='gray')
        return dilated_image

    def find_contours_line(self, image):
        (contours, heirarchy) = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) 
        return sorted_contours_lines

    def find_contours_text(self, image):
        kernel = np.ones((3,15), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations = 1)
        # plt.imshow(dilated2, cmap='gray')
        return dilated

    def get_nth_word(self, n):
        nth_word = self.words_list[n]
        roi_n = self.image[nth_word[1]:nth_word[3], nth_word[0]:nth_word[2]]
        plt.imshow(roi_n)
        return roi_n

    def segmentation(self, image, resize=True):
        # image = cv2.imread(path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resizing image
        if resize:
            image = self.resize_image(image)
            
        self.image = image
        # Thresholding image
        thresholded_image = self.thresholding(image)

        # Dilation
        dilated_image = self.dilation(thresholded_image)

        # Find countours line
        sorted_contours_lines = self.find_contours_line(dilated_image)

        # Find countours word

        image_copy = self.image.copy()
        words_list = []

        dilated = self.find_contours_text(thresholded_image)

        # Draw rectangles

        for line in sorted_contours_lines:
            x, y, w, h = cv2.boundingRect(line)
            roi_line = dilated[y: y+w, x:x+w]

            (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_contour_words = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr))

            for word in sorted_contour_words:
                x2, y2, w2, h2 = cv2.boundingRect(word)
                words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
                cv2.rectangle(image_copy, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (255,255,100),2)

        self.words_list = words_list
        plt.imshow(image_copy)
        return image_copy