import numpy as np
from skimage import io, img_as_float, filters, morphology
from skimage.util import view_as_windows
from sklearn.metrics import mean_squared_error
from skimage.transform import resize
from skimage.color import rgb2gray
from typing import Tuple
import os
from tqdm import tqdm


class ImageQuilting:
    """
    ImageQuilting class implements the image quilting algorithm for texture synthesis
    and inpainting. It can be used to remove and reconstruct parts of an image using
    overlapping patches from the original image.
    """

    def __init__(self, image_path: str, patch_size: int = 10, overlap_size: int = 2):
        """
        Initializes the ImageQuilting instance.
        """
        self.image_path = f"data/benchmark/{image_path}"
        self.patch_size = patch_size
        self.overlap_size = overlap_size
        self.image = self.load_image()
        self.mask = self.generate_mask()
        self.patches = self.extract_patches()

    def load_image(self) -> np.ndarray:
        """
        Loads and returns the input image as a numpy array.
        """
        return img_as_float(io.imread(self.image_path))

    def generate_mask(self) -> np.ndarray:
        """
        Generates and returns a mask for inpainting based on edges and texture variance.
        """
        grayscale_image = rgb2gray(self.image)

        # Edge detection using Sobel filter
        edges = filters.sobel(grayscale_image)

        # Dilate edges to include surrounding areas
        dilated_edges = morphology.dilation(edges, morphology.square(5))

        # Create mask by thresholding
        edge_mask = dilated_edges > 0.1

        # Compute texture variance using Sobel filter
        texture_variance = filters.sobel(grayscale_image)
        texture_variance_mask = texture_variance < 0.01

        # Combine edge mask and texture variance mask
        mask = np.logical_or(edge_mask, texture_variance_mask)

        return mask

    def extract_patches(self) -> np.ndarray:
        """
        Extracts overlapping patches from the input image.
        """
        patches = view_as_windows(self.image, (self.patch_size, self.patch_size, 3))
        return patches.reshape(-1, self.patch_size, self.patch_size, 3)

    def find_best_patch(
        self, target_patch: np.ndarray, overlap_height: int, overlap_width: int
    ) -> np.ndarray:
        """
        Finds the best matching patch for a given target patch based on the overlap area.
        """
        min_error = float("inf")
        best_patch = None

        for patch in self.patches:
            error = np.sum(
                (
                    patch[:overlap_height, :overlap_width, :]
                    - target_patch[:overlap_height, :overlap_width, :]
                )
                ** 2
            )
            if error < min_error:
                min_error = error
                best_patch = patch

        return best_patch

    def stitch_patches(self) -> np.ndarray:
        """
        Stitches patches to fill in the masked area.
        """
        output_image = self.image.copy()
        h, w, _ = self.image.shape

        for i in tqdm(
            range(0, h, self.patch_size - self.overlap_size),
            desc="Stitching patches (rows)",
        ):
            for j in tqdm(
                range(0, w, self.patch_size - self.overlap_size),
                desc="Stitching patches (cols)",
                leave=False,
            ):
                patch_height = min(self.patch_size, h - i)
                patch_width = min(self.patch_size, w - j)
                overlap_height = min(self.overlap_size, patch_height)
                overlap_width = min(self.overlap_size, patch_width)

                if self.mask[i : i + patch_height, j : j + patch_width].any():
                    target_patch = output_image[
                        i : i + patch_height, j : j + patch_width
                    ]
                    best_patch = self.find_best_patch(
                        target_patch, overlap_height, overlap_width
                    )
                    best_patch = best_patch[:patch_height, :patch_width]
                    output_image[i : i + patch_height, j : j + patch_width] = best_patch

        return output_image

    def inpaint(self) -> np.ndarray:
        """
        Inpaints the image using the image quilting method.
        """
        output_image = self.stitch_patches()
        return output_image

    def calculate_mse(
        self, original_image: np.ndarray, inpainted_image: np.ndarray
    ) -> float:
        """
        Calculates and returns the Mean Squared Error (MSE) between the original and inpainted images.
        """
        mask = self.mask.astype(bool)
        mse = mean_squared_error(original_image[mask], inpainted_image[mask])
        return mse

    def save_image(self, output_image: np.ndarray, path: str) -> None:
        """
        Saves the inpainted image to the specified path.
        """
        # Clip values to range [0, 1] and convert to uint8
        output_image_clipped = np.clip(output_image, 0, 1)
        output_image_uint8 = (output_image_clipped * 255).astype(np.uint8)
        io.imsave(path, output_image_uint8)

    def inpaint_and_save_best(
        self,
        patch_sizes: Tuple[int] = (5, 10, 15),
        overlap_sizes: Tuple[int] = (1, 2, 3, 4),
    ) -> None:
        """
        Optimizes patch size and overlap size based on MSE and saves the best inpainted image.
        """
        best_patch_size = None
        best_overlap_size = None
        min_mse = float("inf")
        best_image = None

        for patch_size in tqdm(patch_sizes, desc="Patch sizes"):
            for overlap_size in tqdm(overlap_sizes, desc="Overlap sizes", leave=False):
                self.patch_size = patch_size
                self.overlap_size = overlap_size
                self.patches = self.extract_patches()
                inpainted_image = self.inpaint()
                inpainted_image = resize(
                    inpainted_image, self.image.shape, anti_aliasing=True
                )
                mse = self.calculate_mse(self.image, inpainted_image)

                if mse < min_mse:
                    min_mse = mse
                    best_patch_size = patch_size
                    best_overlap_size = overlap_size
                    best_image = inpainted_image

        print(
            f"Best Patch Size: {best_patch_size}, Best Overlap Size: {best_overlap_size}"
        )
        print(f"Best MSE: {min_mse}")

        if best_image is not None:
            output_file_name = self.get_output_file_name()
            output_path = os.path.join("data", "image-quilting", output_file_name)
            self.save_image(best_image, output_path)

    def get_output_file_name(self) -> str:
        """
        Generates an output file name based on the original file name.
        """
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        return f"{base_name}-reconstruct.jpg"
