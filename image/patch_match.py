import os
import cv2
import numpy as np


class PatchMatch:
    def __init__(self, image: np.ndarray, mask: np.ndarray, patch_size: int = 7):
        """
        Initialize the PatchMatch object.
        """
        self.image = image
        self.mask = mask
        self.patch_size = patch_size
        self.height, self.width, self.channels = image.shape
        self.half_patch = patch_size // 2
        self.offsets = self.initialize_offsets()

    def initialize_offsets(self) -> np.ndarray:
        """
        Initialize the offset map with random values.
        """
        offsets = np.zeros((self.height, self.width, 2), dtype=np.int32)
        for y in range(self.height):
            for x in range(self.width):
                offsets[y, x] = [
                    np.random.randint(-self.width, self.width),
                    np.random.randint(-self.height, self.height),
                ]
        return offsets

    def compute_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Compute the distance between two patches.
        """
        patch1 = self.get_patch(x1, y1)
        patch2 = self.get_patch(x2, y2)
        return np.sum((patch1 - patch2) ** 2)

    def get_patch(self, x: int, y: int) -> np.ndarray:
        """
        Get a patch from the image centered at (x, y).
        """
        x1, x2 = max(0, x - self.half_patch), min(self.width, x + self.half_patch + 1)
        y1, y2 = max(0, y - self.half_patch), min(self.height, y + self.half_patch + 1)

        # Ensure patch size is valid
        patch_width = x2 - x1
        patch_height = y2 - y1
        if patch_width <= 0 or patch_height <= 0:
            return np.zeros(
                (self.patch_size, self.patch_size, self.channels), dtype=np.uint8
            )

        patch = np.zeros(
            (self.patch_size, self.patch_size, self.channels), dtype=np.uint8
        )
        patch[
            (y1 - y + self.half_patch) : (y2 - y + self.half_patch),
            (x1 - x + self.half_patch) : (x2 - x + self.half_patch),
        ] = self.image[y1:y2, x1:x2]
        return patch

    def inpaint(self, iterations: int = 5) -> np.ndarray:
        """
        Inpaint the image using the PatchMatch algorithm.
        """
        for _ in range(iterations):
            for y in range(self.height):
                for x in range(self.width):
                    if self.mask[y, x] == 0:
                        continue
                    current_offset = self.offsets[y, x]
                    best_offset = current_offset
                    best_distance = self.compute_distance(
                        x, y, x + current_offset[0], y + current_offset[1]
                    )
                    for dy in [-1, 1]:
                        for dx in [-1, 1]:
                            nx, ny = (
                                x + current_offset[0] + dx,
                                y + current_offset[1] + dy,
                            )
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                distance = self.compute_distance(x, y, nx, ny)
                                if distance < best_distance:
                                    best_distance = distance
                                    best_offset = [nx - x, ny - y]
                    self.offsets[y, x] = best_offset
        return self.reconstruct_image()

    def reconstruct_image(self) -> np.ndarray:
        """
        Reconstruct the inpainted image using the offset map.
        """
        result = self.image.copy()
        for y in range(self.height):
            for x in range(self.width):
                if self.mask[y, x] == 0:
                    continue
                offset = self.offsets[y, x]
                new_x = x + offset[0]
                new_y = y + offset[1]

                # Ensure new_x and new_y are within image bounds
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    result[y, x] = self.image[new_y, new_x]
                else:
                    # Handle out-of-bound offsets (e.g., use the original pixel or some other strategy)
                    result[y, x] = self.image[
                        y, x
                    ]  # Or use any other fallback strategy

        return result


class PatchMatchImageInpainting:
    """
    Process image inpainting by blind mask using PatchMatch.
    """

    def split_filename(self, file_name: str) -> tuple:
        """
        Split the filename into the name without the extension and the extension.
        """
        file_name_without_extension, file_extension = os.path.splitext(file_name)
        return file_name_without_extension, file_extension

    def find_removable_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Find and return a mask of the removable regions in the image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to make regions more pronounced
        dilated = cv2.dilate(edges, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a mask for the regions to be removed
        mask = np.zeros_like(gray)

        # Draw contours on the mask
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        return mask

    def inpaint_image(self, image_path: str) -> np.ndarray:
        """
        Inpaint the image using PatchMatch.
        """
        # Load image
        image = cv2.imread(image_path)

        # Find removable regions
        mask = self.find_removable_regions(image)

        # Perform inpainting using PatchMatch
        patch_match = PatchMatch(image, mask)
        inpainted_image = patch_match.inpaint()

        # Highlight the removed section in white
        output_image = inpainted_image.copy()
        output_image[mask == 255] = [255, 255, 255]

        return output_image

    def create_mask_from_highlighted(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask from the highlighted regions in the image.
        """
        if image is None:
            raise ValueError("The image is None, please check the image path.")

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold to create a binary mask where white regions are marked
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

        return mask

    def reconstruct_inpaint_image(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct the inpainted image using PatchMatch.
        """
        if image is None:
            raise ValueError("The image is None, please check the image path.")
        if mask is None:
            raise ValueError(
                "The mask is None, please check the mask creation process."
            )

        # Ensure the mask is single-channel and binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)

        # Ensure the mask and image have the same dimensions
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        # Perform inpainting using PatchMatch
        patch_match = PatchMatch(image, mask)
        inpainted_image = patch_match.inpaint()
        return inpainted_image

    def calculate_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between the original and reconstructed images.
        """
        mse = np.mean((original - reconstructed) ** 2)
        return mse

    def reconstruct_image(
        self, original_image: np.ndarray, highlighted_image: np.ndarray
    ) -> tuple:
        """
        Reconstruct the image and calculate the Mean Squared Error (MSE).
        """
        # Create a mask from the highlighted regions
        mask = self.create_mask_from_highlighted(highlighted_image)

        # Perform inpainting to reconstruct the highlighted regions using PatchMatch
        reconstructed_image = self.reconstruct_inpaint_image(original_image, mask)

        # Calculate the MSE between the original and reconstructed images
        mse = self.calculate_mse(original_image, reconstructed_image)

        return reconstructed_image, mse

    def save_output(self, image: np.ndarray, filename: str) -> None:
        """
        Save the output image to a file.
        """
        base_path = "data/patchmatch/"
        full_path = base_path + filename
        cv2.imwrite(full_path, image)
        print(f"Saved: {full_path}")

    def process_inpaint(self, file_name: str) -> None:
        """
        Process the inpainting of the image and save the results.
        """
        file_name_without_extension, _ = self.split_filename(file_name)
        image_path = f"data/benchmark/{file_name}"

        # Inpaint using PatchMatch
        output_patchmatch = self.inpaint_image(image_path)
        self.save_output(
            output_patchmatch, f"{file_name_without_extension}_inpainted_patchmatch.jpg"
        )

    def process_reconstruct(self, file_name: str) -> None:
        """
        Process the reconstruction of the image and save the results.
        """
        file_name_without_extension, _ = self.split_filename(file_name)
        original_image_path = f"data/benchmark/{file_name}"
        highlighted_image_path = (
            f"data/patchmatch/{file_name_without_extension}_inpainted_patchmatch.jpg"
        )

        # Load the original and highlighted images
        original_image = cv2.imread(original_image_path)
        highlighted_image = cv2.imread(highlighted_image_path)

        if original_image is None:
            raise ValueError(
                f"Could not load the original image from path: {original_image_path}"
            )
        if highlighted_image is None:
            raise ValueError(
                f"Could not load the highlighted image from path: {highlighted_image_path}"
            )

        best_mse = float("inf")
        best_reconstruction = None
        best_dilation_size = None

        for dilation_size in range(1, 30):
            # Create a dilated mask to test
            mask = self.create_mask_from_highlighted(highlighted_image)
            kernel = np.ones((dilation_size, dilation_size), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)

            # Perform inpainting using PatchMatch
            reconstructed_image = self.reconstruct_inpaint_image(
                original_image, dilated_mask
            )

            # Calculate MSE
            mse = self.calculate_mse(original_image, reconstructed_image)
            print(f"MSE with dilation size {dilation_size}: {mse}")

            # Update best result
            if mse < best_mse:
                best_mse = mse
                best_reconstruction = reconstructed_image
                best_dilation_size = dilation_size

        # Save the best reconstruction
        self.save_output(
            best_reconstruction,
            f"{file_name_without_extension}_best_reconstruction.jpg",
        )
        print(f"Best dilation size: {best_dilation_size} with MSE: {best_mse}")
