import cv2
import numpy as np
from .image_compression import ImageCompression
from .image_processing import ImageProcessing
from .edge_detection import EdgeDetection


class Inpainting:
    def __init__(self, image: np.ndarray | None = None) -> None:
        """
        Initializes the class instance.
        """
        self.image = image
        self.mask = None

    # Reference: Otsu, N. (1979). "A threshold selection method from gray-level histograms". IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66. DOI:10.1109/TSMC.1979.4310076
    def select_removable_area_by_high_intensity(self, threshold: int = 50) -> tuple:
        """
        Selects areas for inpainting by identifying regions with high similarity.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        img = ImageProcessing()
        gray = img.convert_to_gray(self.image)
        _, self.mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Ensure the mask is 8-bit single-channel
        if self.mask.dtype != np.uint8:
            self.mask = self.mask.astype(np.uint8)

        return self.mask

    # Reference: Forsyth, D. A., & Ponce, J. (2002). "Computer Vision: A Modern Approach". Prentice Hall. Book
    def select_removable_area_by_high_intensity_and_edge(
        self, threshold: int = 50, edge_method: str = "canny"
    ) -> tuple:
        """
        Selects areas for inpainting by identifying regions with high similarity and edges.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        # Apply edge detection based on the specified method
        edge_detector = EdgeDetection(self.image)
        if edge_method not in [
            "sobel",
            "prewitt",
            "roberts",
            "log",
            "canny",
            "scharr",
            "zero_crossing",
            "optimal_canny",
        ]:
            raise ValueError(
                f"Invalid edge detection method specified. Choose from 'sobel', 'prewitt', 'roberts', 'log', 'canny', 'scharr', 'zero_crossing', 'optimal_canny'."
            )

        edge_func = getattr(edge_detector, edge_method)
        edges = edge_func()

        # Combine the edges with the original mask (if available)
        if self.mask is not None:
            combined_mask = np.maximum(self.mask, edges)
            _, self.mask = cv2.threshold(
                combined_mask, threshold, 255, cv2.THRESH_BINARY
            )
        else:
            _, self.mask = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)

        # Ensure the mask is 8-bit single-channel
        if self.mask.dtype != np.uint8:
            self.mask = self.mask.astype(np.uint8)

        return self.mask

    def inpaint_image(self, method: str = "telea", radius: int = 3) -> np.ndarray:
        """
        Inpaints the selected areas in the image using the specified method and radius.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")
        elif self.mask is None:
            raise ValueError("Mask is not available for this action.")
        elif not isinstance(self.mask, np.ndarray):
            raise ValueError("Mask is not a numpy array.")

        # Ensure the mask is an 8-bit single-channel image
        if self.mask.dtype != np.uint8:
            self.mask = self.mask.astype(np.uint8)

        # Ensure the mask is the same size as the image
        if self.image.shape[:2] != self.mask.shape:
            self.mask = cv2.resize(
                self.mask, (self.image.shape[1], self.image.shape[0])
            )

        if method == "telea":
            inpainted_image = cv2.inpaint(
                self.image, self.mask, radius, cv2.INPAINT_TELEA
            )
        elif method == "ns":
            inpainted_image = cv2.inpaint(self.image, self.mask, radius, cv2.INPAINT_NS)
        else:
            raise ValueError(
                "Invalid inpainting method specified. Choose 'telea' or 'ns'."
            )

        return inpainted_image

    def compress_remaining_areas(self, quality: int = 80) -> np.ndarray:
        """
        Compresses the remaining parts of the image that can't be restored by inpainting.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        compressor = ImageCompression(self.image)
        compressed_image = compressor.compress_jpeg(quality=quality)
        return compressed_image

    def generate_metadata(
        self, inpainted_image: object, compressed_image_data: np.ndarray
    ) -> dict:
        """
        Generates metadata about the inpainted and compressed areas.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        metadata = {
            "original_shape": self.image.shape,
            "inpainted_shape": inpainted_image.shape,
            "compressed_size": len(compressed_image_data),
            "mask_shape": self.mask.shape,
            "mask_non_zero_count": np.count_nonzero(self.mask),
            "compressed_data": compressed_image_data,
        }

        return metadata

    def custom_compression(
        self,
        quality: int = 80,
        threshold: int = 50,
        method: str = "high_intensity",
        inpaint_method: str = "telea",
        inpaint_radius: int = 3,
    ) -> dict:
        """
        Custom compression algorithm combining inpainting and image compression.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")

        # Step 1: Select area for removal based on the specified method
        if method == "high_intensity":
            self.select_removable_area_by_high_intensity(threshold)
        elif method == "high_intensity_and_edge":
            self.select_removable_area_by_high_intensity_and_edge(threshold)
        else:
            raise ValueError(
                "Invalid method specified. Choose 'high_intensity' or 'high_intensity_and_edge'."
            )

        # Step 2: Inpaint the selected areas
        inpainted_image = self.inpaint_image(inpaint_method, inpaint_radius)

        # Step 3: Compress the remaining areas
        compressed_image = self.compress_remaining_areas(quality)

        # Step 4: Generate metadata
        metadata = self.generate_metadata(inpainted_image, compressed_image)

        return metadata

    def reconstruct_mask(
        self, mask_shape: tuple, mask_non_zero_count: int
    ) -> np.ndarray:
        """
        Reconstructs a mask based on the given shape and non-zero count.
        """
        mask = np.zeros(mask_shape, dtype=np.uint8)
        non_zero_indices = np.random.choice(
            np.prod(mask.shape), mask_non_zero_count, replace=False
        )
        np.put(mask, non_zero_indices, 255)
        return mask

    def restore_image(
        self, metadata: dict, inpaint_method: str = "telea", inpaint_radius: int = 3
    ) -> object:
        """
        Restores the original image from the compressed and inpainted data.
        """
        compressor = ImageCompression(self.image)
        decompressed_image = compressor.decompress_jpeg(metadata["compressed_data"])
        mask = self.reconstruct_mask(
            metadata["mask_shape"], metadata["mask_non_zero_count"]
        )
        if inpaint_method == "telea":
            restored_image = cv2.inpaint(
                decompressed_image, mask, inpaint_radius, cv2.INPAINT_TELEA
            )
        elif inpaint_method == "ns":
            restored_image = cv2.inpaint(
                decompressed_image, mask, inpaint_radius, cv2.INPAINT_NS
            )
        else:
            raise ValueError(
                "Invalid inpainting method specified. Choose 'telea' or 'ns'."
            )
        return restored_image

    def apply_mask_and_set_white(self, mask: np.ndarray, output_path: str) -> None:
        """
        Apply the mask to the original image, set the black sections in the mask to white in the original image,
        and save the resulting image.
        """
        if self.image is None:
            raise ValueError("Image is not available for this action.")
        if mask is None:
            raise ValueError("Mask is not available for this action.")
        if not isinstance(mask, np.ndarray):
            raise ValueError("Mask is not a numpy array.")

        # Ensure the mask is 8-bit single-channel
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Ensure the mask is the same size as the image
        if self.image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]))

        # Create a copy of the original image to modify
        result_image = self.image.copy()

        # Set the black areas in the mask to white in the result image
        result_image[mask == 0] = [255, 255, 255]

        # Save the resulting image
        cv2.imwrite(output_path, result_image)
