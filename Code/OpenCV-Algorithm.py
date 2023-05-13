import cv2
import numpy as np

class SegmentationMap:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.segmentation_map = None

    def generate_segmentation_map(self):
    # Convert image to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank image for the segmentation map
        segmentation_map = np.zeros_like(gray_image)

        # Draw contours on the segmentation map
        for i, contour in enumerate(contours):
            cv2.drawContours(segmentation_map, contours, i, (i+1), -1)

        self.segmentation_map = segmentation_map

    def save_segmentation_map(self, file_name):
        cv2.imwrite(file_name, self.segmentation_map)

# Example usage
segmentation_map = SegmentationMap("../Dataset/l1.png")
segmentation_map.generate_segmentation_map()
segmentation_map.save_segmentation_map("../Dataset/segmentation_map.jpg")
