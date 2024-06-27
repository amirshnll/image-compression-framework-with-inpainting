import os
import cv2
import numpy as np


# Reference: Pérez, P., Gangnet, M., & Blake, A. (2003). "Poisson Image Editing." ACM Transactions on Graphics (TOG), 22(3), 313-318.
class PoissonImageEditing:
    """
    Process image reconstruction using Poisson Image Editing
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

    def find_center(self, mask: np.ndarray) -> tuple:
        """
        Find the center of the largest contour in the mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0
        return (center_x, center_y)

    def adjust_center(self, center, target_shape, mask_shape):
        """
        Adjust the center to ensure it's within the bounds of the target image.
        """
        center_x, center_y = center
        mask_h, mask_w = mask_shape[:2]
        target_h, target_w = target_shape[:2]

        center_x = min(max(center_x, mask_w // 2), target_w - mask_w // 2)
        center_y = min(max(center_y, mask_h // 2), target_h - mask_h // 2)
        return (center_x, center_y)

    def poisson_edit(self, source, target, mask, center):
        """
        Apply Poisson Image Editing to blend the source image into the target image using the provided mask.
        """
        # Ensure mask is 3-channel
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Ensure the mask size matches the target size
        if mask.shape[:2] != target.shape[:2]:
            mask = cv2.resize(
                mask,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Adjust center to be within bounds
        center = self.adjust_center(center, target.shape, mask.shape)

        # Debugging statements
        print(f"Source shape: {source.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Center: {center}")

        mixed_clone = cv2.seamlessClone(source, target, mask, center, cv2.NORMAL_CLONE)
        return mixed_clone

    def reconstruct_image(
        self, original_image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct the image using Poisson Image Editing with the given mask.
        """
        if original_image is None:
            raise ValueError("The original image is None, please check the image path.")
        if mask is None:
            raise ValueError(
                "The mask is None, please check the mask creation process."
            )

        # Ensure the mask and image have the same dimensions
        if mask.shape != original_image.shape[:2]:
            mask = cv2.resize(
                mask,
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Find the center for blending
        center = self.find_center(mask)

        # Create source image by copying the original image
        source_image = original_image.copy()

        # Perform Poisson Image Editing
        reconstructed_image = self.poisson_edit(
            source_image, original_image, mask, center
        )
        return reconstructed_image

    def save_output(self, image: np.ndarray, filename: str) -> None:
        """
        Save the output image to a file.
        """
        base_path = "data/poisson/"
        os.makedirs(base_path, exist_ok=True)
        full_path = base_path + filename
        cv2.imwrite(full_path, image)
        print(f"Saved: {full_path}")

    def calculate_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between the original image and the reconstructed image.
        """
        mse = np.mean((original - reconstructed) ** 2)
        return mse

    def process_poisson_edit(self, file_name: str) -> None:
        """
        Process the Poisson Image Editing of the image and save the results.
        """
        file_name_without_extension, _ = self.split_filename(file_name)
        image_path = f"data/benchmark/{file_name}"

        # Load image
        original_image = cv2.imread(image_path)

        # Find removable regions in the image
        mask = self.find_removable_regions(original_image)

        # Perform Poisson Image Editing
        output_image = self.reconstruct_image(original_image, mask)
        self.save_output(
            output_image, f"{file_name_without_extension}_poisson_edited.jpg"
        )

        # Calculate MSE
        mse = self.calculate_mse(original_image, output_image)
        print(f"MSE between original and output image: {mse}")