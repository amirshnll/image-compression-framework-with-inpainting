import os
import cv2
import numpy as np


class TextureSynthesis:
    """
    Performs texture synthesis using the PatchMatch algorithm to fill masked regions in images, enhancing inpainting capabilities.
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
        if image is None:
            raise ValueError("Invalid image provided for finding removable regions.")

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

    def save_mask_to_file(self, mask: np.ndarray, filename: str) -> None:
        """
        Save the mask to a text file.
        """
        np.savetxt(filename, mask, fmt="%d")
        print(f"Saved mask to {filename}")

    def load_mask_from_file(self, filename: str) -> np.ndarray:
        """
        Load the mask from a text file.
        """
        return np.loadtxt(filename, dtype=np.uint8)

    def calculate_mask(self, image_path: str, mask_filename: str) -> np.ndarray:
        """
        Calculate and return the mask for the given image path.
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at: {image_path}")

        # Find removable regions
        mask = self.find_removable_regions(image)

        # Save the mask to a text file
        self.save_mask_to_file(mask, f"data/texture-synthesis/{mask_filename}-mask.txt")

        return mask

    def texture_synthesis(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform texture synthesis to fill in the masked regions of the image.
        """
        if image is None:
            raise ValueError("Invalid image provided for texture synthesis.")
        if mask is None:
            raise ValueError("Invalid mask provided for texture synthesis.")

        # Prepare the mask (binary, single-channel)
        mask = (mask > 0).astype(np.uint8)

        # Perform inpainting using OpenCV's inpaint function
        inpainted_image = cv2.inpaint(
            image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
        )

        return inpainted_image

    def save_output(self, image: np.ndarray, filename: str) -> None:
        """
        Save the output image to a file.
        """
        base_path = "data/texture-synthesis/"
        full_path = os.path.join(base_path, filename)
        cv2.imwrite(full_path, image)
        print(f"Saved: {full_path}")

    def calculate_mse(
        self, original_image: np.ndarray, synthesized_image: np.ndarray
    ) -> float:
        """
        Calculate the Mean Squared Error between the original image and the synthesized image.
        """
        if original_image.shape != synthesized_image.shape:
            raise ValueError(
                "Original image and synthesized image must have the same dimensions."
            )

        mse = np.mean((original_image - synthesized_image) ** 2)
        return mse

    def process_texture_synthesis(self, image_file: str) -> None:
        """
        Process texture synthesis for the image, calculate the mask, and save the results.
        """
        try:
            image_folder = "data/benchmark/"
            image_path = image_folder + image_file
            # Calculate mask
            file_name_without_extension, _ = self.split_filename(image_file)
            mask = self.calculate_mask(image_path, file_name_without_extension)

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image at: {image_path}")

            # Perform texture synthesis
            synthesized_image = self.texture_synthesis(image, mask)

            # Save the synthesized image
            output_filename = (
                f"{os.path.splitext(os.path.basename(image_path))[0]}_synthesized.jpg"
            )
            self.save_output(synthesized_image, output_filename)

            # Calculate MSE
            mse = self.calculate_mse(image, synthesized_image)
            print(f"Mean Squared Error between original and synthesized image: {mse}")

            print(f"Texture synthesis completed and saved as {output_filename}")

        except Exception as e:
            print(f"Error processing texture synthesis: {e}")
            raise
