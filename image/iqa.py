import zlib
import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as rmse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from .image_processing import ImageProcessing
from .edge_detection import EdgeDetection


class IQA:
    # IQA class for calculating various metrics used in Image Quality Assessment (IQA)
    def __init__(self, image1: object, image2: object) -> None:
        """
        Initializes the class instance.
        """
        self.image1 = image1
        self.image2 = image2

        # Check if one or both images are None and raise an error if so
        if self.image1 is None or self.image2 is None:
            raise ValueError("One or both images are None.")

        # Check if input images have the same dimensions and raise an error if not
        if self.image1.shape != self.image2.shape:
            raise ValueError("Input images must have the same dimensions.")

    # Calculate Mean Squared Error (MSE) between two input images
    # Reference: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE transactions on image processing, 13(4), 600-612.
    def calculate_mse(self) -> float:
        return mse(self.image1, self.image2)

    # Calculate Peak Signal-to-Noise Ratio (PSNR) based on MSE value
    # Reference: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE transactions on image processing, 13(4), 600-612.
    def calculate_psnr(self, mse: float) -> float:
        if mse == 0:
            return float("inf")
        return psnr(self.image1, self.image2)

    # Calculate Structural Similarity Index Measure (SSIM) between two input images
    # Reference: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE transactions on image processing, 13(4), 600-612.
    def calculate_ssim(self) -> float:
        img = ImageProcessing()
        gray_image1 = img.convert_to_gray(self.image1)
        gray_image2 = img.convert_to_gray(self.image2)
        return ssim(gray_image1, gray_image2)

    # Calculate Root Mean Squared Error (RMSE) between two input images
    # Reference: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE transactions on image processing, 13(4), 600-612.
    def calculate_rmse(self) -> float:
        return rmse(self.image1, self.image2)

    # Mean Absolute Error (MAE) calculation
    # Reference: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE transactions on image processing, 13(4), 600-612.
    def calculate_mae(self) -> float:
        """
        Calculate the Mean Absolute Error (MAE) between two images.

        :return: The MAE value.
        """
        return np.mean(np.abs(self.image1 - self.image2))

    # Entropy calculation
    # Reference: Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal, 27(3), 379-423.
    def calculate_entropy(self) -> tuple:
        """
        Calculate the entropy of two images.

        :return: A tuple containing the entropies of both images.
        """
        img = ImageProcessing()
        gray_image1 = img.convert_to_gray(self.image1)
        gray_image2 = img.convert_to_gray(self.image2)
        hist1, _ = np.histogram(gray_image1.flatten(), bins=256)
        hist2, _ = np.histogram(gray_image2.flatten(), bins=256)
        entropy1 = entropy(hist1)
        entropy2 = entropy(hist2)
        return entropy1, entropy2

    # Compression ratio calculation
    # Reference: Salomon, D., & Motta, G. (2010). Data Compression: The Complete Reference. Springer.
    def calculate_compression_ratio(self, compressed_image: object) -> float:
        """
        Calculate the compression ratio of two images.

        :param compressed_image: The compressed image.
        :return: The compression ratio.
        """
        original_size = self.image1.size + self.image2.size
        compressed_size = compressed_image.size
        return original_size / compressed_size

    # Bitrate calculation
    # Reference: Richardson, I. (2003). H.264 and MPEG-4 Video Compression: Video Coding for Next-generation Multimedia. John Wiley & Sons.
    def calculate_bitrate(self, compressed_image: object) -> float:
        """
        Calculate the bitrate of two images.

        :param compressed_image: The compressed image.
        :return: The bitrate.
        """
        compressed_size = compressed_image.size
        return compressed_size / (self.image1.size + self.image2.size)

    # FSIM calculation
    # Reference: Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). FSIM: A feature similarity index for image quality assessment. IEEE transactions on Image Processing, 20(8), 2378-2386.
    def calculate_fsim(self) -> float:
        """
        Calculate the Feature Similarity Index (FSIM) between two images.

        :return: The FSIM value.
        """
        edge_detector = EdgeDetection()
        img = ImageProcessing()
        original = self.image1.astype(np.float32)
        compressed = self.image2.astype(np.float32)
        grad_original_x = edge_detector.apply_sobel_operator(original, 1, 0, 3)
        grad_original_y = edge_detector.apply_sobel_operator(original, 0, 1, 3)
        magnitude_original = img.calculate_magnitude(grad_original_x, grad_original_y)
        grad_compressed_x = edge_detector.apply_sobel_operator(compressed, 1, 0, 3)
        grad_compressed_y = edge_detector.apply_sobel_operator(compressed, 0, 1, 3)
        magnitude_compressed = img.calculate_magnitude(
            grad_compressed_x, grad_compressed_y
        )
        T = 0.85
        pc_original = magnitude_original > (T * np.max(magnitude_original))
        pc_compressed = magnitude_compressed > (T * np.max(magnitude_compressed))
        pc_sum = np.sum(pc_original.astype(float) + pc_compressed.astype(float))
        pc_product = np.sum(pc_original.astype(float) * pc_compressed.astype(float))
        fsim_score = pc_product / pc_sum if pc_sum != 0 else 0
        return fsim_score

    # NCD calculation
    # Reference: Li, M., Chen, X., Li, X., Ma, B., & Vitányi, P. M. (2004). The similarity metric. IEEE Transactions on Information Theory, 50(12), 3250-3264.
    def calculate_ncd(self, image1_path: str, image2_path: str) -> float:
        """
        Calculate the Normalized Compression Distance (NCD) between two images.

        :param image1_path: The path to the first image.
        :param image2_path: The path to the second image.
        :return: The NCD value.
        """
        image1 = open(image1_path, "rb").read()
        image2 = open(image2_path, "rb").read()
        compressed_size1 = len(zlib.compress(image1))
        compressed_size2 = len(zlib.compress(image2))
        ncd = (
            max(compressed_size1, compressed_size2)
            - min(compressed_size1, compressed_size2)
        ) / max(compressed_size1, compressed_size2)
        return ncd
