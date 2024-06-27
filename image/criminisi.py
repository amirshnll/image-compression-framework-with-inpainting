import os
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


class CriminisiImageInpainting:
    """
    Process image inpainting using the Criminisi Algorithm.
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

    def compute_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the gradient magnitude of the image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients along the x and y axis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)

        # Convert to 8-bit image
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

        return gradient_magnitude

    def compute_patch_priority(
        self, mask: np.ndarray, gradient_magnitude: np.ndarray
    ) -> np.ndarray:
        """
        Compute the patch priority based on the mask and gradient magnitude.
        """
        # Compute distance transform
        distance_transform = distance_transform_edt(mask)

        # Normalize the distance transform
        distance_transform = distance_transform / distance_transform.max()

        # Convert gradient magnitude to grayscale if it's not already
        if len(gradient_magnitude.shape) == 3:
            gradient_magnitude = cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2GRAY)

        # Normalize the gradient magnitude
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()

        # Compute priority as the product of distance transform and gradient magnitude
        priority = distance_transform * gradient_magnitude

        return priority

    def find_highest_priority_patch(self, priority: np.ndarray) -> tuple:
        """
        Find the coordinates of the highest priority patch.
        """
        y, x = np.unravel_index(np.argmax(priority), priority.shape)
        return (x, y)

    def criminisi_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the image using the Criminisi Algorithm.
        """
        # Compute gradient magnitude
        gradient_magnitude = self.compute_gradient_magnitude(image)

        # Compute patch priority
        priority = self.compute_patch_priority(mask, gradient_magnitude)

        # Find the highest priority patch
        patch_center = self.find_highest_priority_patch(priority)

        # Debugging statements
        print(f"Gradient Magnitude shape: {gradient_magnitude.shape}")
        print(f"Priority shape: {priority.shape}")
        print(f"Patch center: {patch_center}")

        # Perform inpainting (example using OpenCV's inpaint function for simplicity)
        inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        return inpainted_image

    def save_output(self, image: np.ndarray, filename: str) -> None:
        """
        Save the output image to a file.
        """
        base_path = "data/criminisi/"
        os.makedirs(base_path, exist_ok=True)
        full_path = base_path + filename
        cv2.imwrite(full_path, image)
        print(f"Saved: {full_path}")

    def process_criminisi_edit(self, file_name: str) -> None:
        """
        Process the Criminisi Image Editing of the image and save the results.
        """
        file_name_without_extension, _ = self.split_filename(file_name)
        image_path = f"data/benchmark/{file_name}"

        # Load image
        original_image = cv2.imread(image_path)

        # Find removable regions in the image
        mask = self.find_removable_regions(original_image)

        # Perform Criminisi Image Editing
        output_image = self.criminisi_inpaint(original_image, mask)
        self.save_output(
            output_image, f"{file_name_without_extension}_criminisi_edited.jpg"
        )
