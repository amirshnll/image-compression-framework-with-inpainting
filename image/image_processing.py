import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from typing import Union


class ImageProcessing:
    """
    This class represents an image and provides methods to open, read, convert to gray scale,
    check if the image is in RGB format, calculate histogram, resize, crop,
    apply Sobel operator, calculate magnitude of gradient.
    """

    def __init__(self, file_path=None) -> None:
        """
        Initializes the class instance.
        """
        self.file_path = file_path
        self.image = None

    def open(self, file_path: str) -> object:
        """
        Opens an image from a file and stores it in the class instance.
        """
        import cv2

        if file_path.lower().endswith(".avif"):
            with Image.open(file_path) as img:
                self.image = np.array(img)
        else:
            self.file_path = file_path
            self.image = cv2.imread(file_path)

        if self.image is None:
            raise ValueError("Image not loaded. Please check the file path.")
        return self.image

    def read(self) -> object:
        """
        Returns the image stored in the class instance.
        """
        import cv2

        if self.image is None:
            raise ValueError("No image loaded. Use the open method first.")
        return self.image

    def rgb_to_gray(self) -> object:
        """
        Converts the image to grayscale format.
        """
        import cv2

        if self.image is None:
            raise ValueError("No image loaded. Use the open method first.")
        img = ImageProcessing()
        return img.convert_to_gray(self.image)

    def is_rgb(self) -> bool:
        """
        Checks if the stored image is in RGB format.
        """
        import cv2

        if self.image is None:
            raise ValueError("No image loaded. Use the open method first.")
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            return True
        else:
            return False

    def histogram(self) -> None:
        """
        Calculates and displays the histogram of the stored image.
        """
        import cv2

        if self.image is None:
            raise ValueError("No image loaded. Use the open method first.")
        if self.is_rgb():
            color = ("b", "g", "r")
            for i, col in enumerate(color):
                histr = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
            plt.show()
        else:
            histr = cv2.calcHist([self.image], [0], None, [256], [0, 256])
            plt.plot(histr)
            plt.xlim([0, 256])
            plt.show()

    def resize(self, width: int, height: int) -> object:
        """
        Resizes the stored image to a given width and height.
        """
        import cv2

        if self.image is None:
            raise ValueError("No image loaded. Use the open method first.")
        return cv2.resize(self.image, (width, height))

    def crop(self, x: int, y: int, width: int, height: int) -> object:
        """
        Crops a region of the stored image.
        """
        import cv2

        if self.image is None:
            raise ValueError("No image loaded. Use the open method first.")
        return self.image[y : y + height, x : x + width]

    def convert_to_gray(self, data: object) -> object:
        """
        Convert an image from BGR color space to grayscale.
        """
        import cv2

        if data.dtype != np.uint8 and data.dtype != np.uint16:
            data = cv2.convertScaleAbs(data)
        return cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    def convert_to_rgb(self, data: object) -> object:
        """
        Convert an image from BGR color space to RGB.
        """
        import cv2

        return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    def save_image(self, output_image: np.ndarray, file_name: str) -> None:
        """
        Save an image.
        """
        import cv2

        cv2.imwrite(file_name, output_image)

    def calculate_magnitude(self, grad_x: np.ndarray, grad_y: np.ndarray) -> object:
        """
        Calculate the magnitude of the gradient.
        """
        import cv2

        return cv2.magnitude(grad_x, grad_y)

    def highlight_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Highlights an image based on the mask values.
        Highlights in red for mask value 255 and green for mask value 0.
        """
        # Convert image to PIL format
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask).convert("L")

        # Create a drawing context
        draw = ImageDraw.Draw(image_pil, "RGBA")

        # Get the dimensions of the image
        width, height = image_pil.size

        # Iterate over each pixel in the mask
        for x in range(width):
            for y in range(height):
                mask_pixel = mask_pil.getpixel((x, y))

                if mask_pixel == 0:
                    draw.rectangle(
                        [x, y, x + 1, y + 1], fill=(0, 255, 0, 80)
                    )  # Green with some transparency
                elif mask_pixel == 255:
                    draw.rectangle(
                        [x, y, x + 1, y + 1], fill=(255, 0, 0, 80)
                    )  # Red with some transparency

        # Convert the image back to numpy array
        highlighted_image = np.array(image_pil)
        return highlighted_image
