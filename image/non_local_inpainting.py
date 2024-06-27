import os
import cv2
import numpy as np


# Reference: Bousseau, A., Paris, S. and Durand, F., 2009. Non-local Means Inpainting. IEEE Transactions on Image Processing, 19(2), pp.287-297. Available at: http://doi.org/10.1109/TIP.2009.203281
class NonLocalMeansInpainting:
    """
    Performing non-local means inpainting on images.
    The NonLocalMeansInpainting class provides a comprehensive framework for performing
    image inpainting, which involves removing unwanted regions from an image and
    reconstructing those regions in a visually plausible manner. This class supports
    multiple inpainting techniques, including the Non-local Means Inpainting method.
    """

    # Define class constants for inpainting methods
    INPAINT_TELEA = cv2.INPAINT_TELEA
    INPAINT_NS = cv2.INPAINT_NS
    INPAINT_NLM = 3  # Assuming you have a custom implementation for Non-local Means

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

    def inpaint_image(
        self, image_path: str, method: int, mask_filename: str
    ) -> np.ndarray:
        """
        Inpaint the image using the specified method and save the mask to a text file.
        """
        # Load image
        image = cv2.imread(image_path)

        # Find removable regions
        mask = self.find_removable_regions(image)

        # Save the mask to a text file
        self.save_mask_to_file(mask, mask_filename)

        # Perform inpainting
        if method in [self.INPAINT_TELEA, self.INPAINT_NS]:
            inpainted_image = cv2.inpaint(image, mask, 3, method)
        elif method == self.INPAINT_NLM:
            inpainted_image = self.inpaint_nlm(image, mask)
        else:
            raise ValueError("Unsupported inpainting method")

        # Highlight the removed section in white
        output_image = inpainted_image.copy()
        output_image[mask == 255] = [255, 255, 255]

        return output_image

    def inpaint_nlm(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Convert mask to 8-bit binary
        mask = (mask > 0).astype(np.uint8)

        # Inpaint using Non-local Means
        # First, perform inpainting on the edges
        inpainted_edges = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

        # Then, perform inpainting using Non-local Means
        inpainted_image = cv2.fastNlMeansDenoisingColored(
            inpainted_edges,
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        return inpainted_image

    def calculate_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between the original and reconstructed images.
        """
        mse = np.mean((original - reconstructed) ** 2)
        return mse

    def process_inpaint(self, file_name: str) -> None:
        """
        Process the inpainting of the image and save the results.
        """
        file_name_without_extension, _ = self.split_filename(file_name)
        image_path = f"data/benchmark/{file_name}"
        mask_filename = f"data/non-local-means/{file_name_without_extension}-mask.txt"

        # Inpaint using Telea's method
        output_telea = self.inpaint_image(image_path, self.INPAINT_TELEA, mask_filename)
        self.save_output(
            output_telea, f"{file_name_without_extension}_inpainted_telea.jpg"
        )

        # Inpaint using Navier-Stokes based method
        output_ns = self.inpaint_image(image_path, self.INPAINT_NS, mask_filename)
        self.save_output(output_ns, f"{file_name_without_extension}_inpainted_ns.jpg")

        # Inpaint using Non-local Means (custom implementation)
        output_nlm = self.inpaint_image(image_path, self.INPAINT_NLM, mask_filename)
        self.save_output(output_nlm, f"{file_name_without_extension}_inpainted_nlm.jpg")

    def save_output(self, image: np.ndarray, filename: str) -> None:
        """
        Save the output image to a file.
        """
        base_path = "data/non-local-means/"
        full_path = os.path.join(base_path, filename)
        cv2.imwrite(full_path, image)
        print(f"Saved: {full_path}")

    def reconstruct_inpaint_image(
        self, image: np.ndarray, mask: np.ndarray, method: int
    ) -> np.ndarray:
        """
        Reconstruct the inpainted image using the given mask and method.
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

        # Perform inpainting
        if method in [self.INPAINT_TELEA, self.INPAINT_NS]:
            inpainted_image = cv2.inpaint(image, mask, 3, method)
        elif method == self.INPAINT_NLM:
            inpainted_image = self.inpaint_nlm(image, mask)
        else:
            raise ValueError("Unsupported inpainting method")

        return inpainted_image

    def process_reconstruct(self, file_name: str) -> None:
        """
        Process the reconstruction of the image and save the results.
        """
        file_name_without_extension, _ = self.split_filename(file_name)
        original_image_path = f"data/benchmark/{file_name}"
        mask_filename = f"data/non-local-means/{file_name_without_extension}-mask.txt"

        # Load the original image
        original_image = cv2.imread(original_image_path)

        # Load the mask from the text file
        mask = self.load_mask_from_file(mask_filename)

        if original_image is None:
            raise ValueError(
                f"Could not load the original image from path: {original_image_path}"
            )

        methods = [cv2.INPAINT_TELEA, cv2.INPAINT_NS, self.INPAINT_NLM]
        method_names = ["Telea", "Navier-Stokes", "Non-local Means"]
        best_mse = float("inf")
        best_reconstruction = None
        best_method = None

        for method, method_name in zip(methods, method_names):
            for dilation_size in range(1, 30):
                # Create a dilated mask to test
                kernel = np.ones((dilation_size, dilation_size), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=1)

                # Perform inpainting
                reconstructed_image = self.reconstruct_inpaint_image(
                    original_image, dilated_mask, method
                )

                # Calculate MSE
                mse = self.calculate_mse(original_image, reconstructed_image)
                print(
                    f"MSE for method {method_name} with dilation size {dilation_size}: {mse}"
                )

                # Update best result
                if mse < best_mse:
                    best_mse = mse
                    best_reconstruction = reconstructed_image
                    best_method = f"{method_name} with dilation size {dilation_size}"

        # Save the best reconstruction
        self.save_output(
            best_reconstruction,
            f"{file_name_without_extension}-best-reconstruction.jpg",
        )
        print(f"Best method: {best_method} with MSE: {best_mse}")
