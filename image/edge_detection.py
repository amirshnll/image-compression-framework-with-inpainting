import cv2
import numpy as np
from scipy import ndimage
from .image_processing import ImageProcessing


class EdgeDetection:
    def __init__(self, image: np.ndarray | None = None) -> None:
        """
        Initializes the class instance.
        """
        if image is not None:
            img = ImageProcessing()
            self.image = image
            if len(self.image.shape) == 3:
                self.image = img.convert_to_gray(self.image)

    # 1. Sobel Edge Detection
    # Reference: Sobel, I., & Feldman, G. (1968). A 3x3 Isotropic Gradient Operator for Image Processing.
    def sobel(self) -> np.ndarray:
        """
        Applies Sobel edge detection algorithm to the input image.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        sobelx = self.apply_sobel_operator(self.image, 1, 0)
        sobely = self.apply_sobel_operator(self.image, 0, 1)
        sobel = np.hypot(sobelx, sobely)
        return sobel

    def apply_sobel_operator(
        self,
        image: np.ndarray,
        dx: int,
        dy: int,
        ksize: int = 3,
        ddepth: int = cv2.CV_64F,
    ) -> object:
        """
        Apply Sobel operator to an image.
        """
        return cv2.Sobel(image, ddepth, dx, dy, ksize)

    # 2. Prewitt Edge Detection
    # Reference: Prewitt, J. M. S. (1970). Object Enhancement and Extraction. Picture Processing and Psychopictorics.
    def prewitt(self) -> np.ndarray:
        """
        Applies Prewitt edge detection algorithm to the input image.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = ndimage.convolve(self.image, kernelx)
        prewitty = ndimage.convolve(self.image, kernely)
        prewitt = np.hypot(prewittx, prewitty)
        prewitt = (prewitt / prewitt.max() * 255).astype(np.uint8)
        return prewitt

    # 3. Roberts Edge Detection
    # Reference: Roberts, L. G. (1963). Machine Perception of Three-Dimensional Solids.
    def roberts(self) -> np.ndarray:
        """
        Applies Roberts edge detection algorithm to the input image.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = ndimage.convolve(self.image, kernelx)
        robertsy = ndimage.convolve(self.image, kernely)
        roberts = np.hypot(robertsx, robertsy)
        roberts = (roberts / roberts.max() * 255).astype(np.uint8)
        return roberts

    # 4. Laplacian of Gaussian (LoG)
    # Reference: Marr, D., & Hildreth, E. (1980). Theory of Edge Detection. Proceedings of the Royal Society B.
    def log(self) -> object:
        """
        Applies Laplacian of Gaussian edge detection algorithm to the input image.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        log = self.apply_laplacian_operator(self.image)
        return log

    def apply_laplacian_operator(
        self,
        image: np.ndarray,
        ddepth: int = cv2.CV_64F,
    ) -> object:
        """
        Apply Sobel operator to an image.
        """
        return cv2.Laplacian(image, ddepth)

    # 5. Canny Edge Detection
    # Reference: Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence.
    def canny(self, low_threshold=100, high_threshold=200):
        """
        Applies Canny edge detection algorithm to the input image with optional low and high threshold values.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        return cv2.Canny(self.image, low_threshold, high_threshold)

    # 6. Scharr Edge Detection
    # Reference: Scharr, H. (2000). Optimal Operators in Digital Image Processing.
    def scharr(self) -> np.ndarray:
        """
        Applies Scharr edge detection algorithm to the input image.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        scharrx = self.apply_scharr_operator(self.image, 1, 0)
        scharry = self.apply_scharr_operator(self.image, 0, 1)
        scharr = np.hypot(scharrx, scharry)
        return scharr

    def apply_scharr_operator(
        self,
        image: np.ndarray,
        dx: int,
        dy: int,
        ddepth: int = cv2.CV_64F,
    ) -> object:
        """
        Apply Sobel operator to an image.
        """
        return cv2.Scharr(image, ddepth, dx, dy)

    # 7. Marr-Hildreth (Zero-Crossing)
    # Reference: Marr, D., & Hildreth, E. (1980). Theory of Edge Detection. Proceedings of the Royal Society B.
    def zero_crossing(self) -> np.ndarray:
        """
        Applies Marr-Hildreth edge detection algorithm to the input image using zero-crossing method.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        log = ndimage.gaussian_laplace(self.image, sigma=1)
        zero_crossing = np.zeros_like(log)
        zero_crossing[np.where(np.diff(np.sign(log)))[0]] = 255
        return zero_crossing

    # 8. Apply Canny edge detection using optimal values for low and high thresholds.
    # Reference: Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence.
    def optimal_canny(self) -> object:
        """
        Applies Canny edge detection algorithm to the input image with optimal threshold values calculated based on median intensity of the image.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        v = np.median(self.image)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = self.canny(lower, upper)
        return edged
