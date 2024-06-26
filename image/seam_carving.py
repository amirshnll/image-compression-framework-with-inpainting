import os
import cv2
import numpy as np


class ImageSeamCarving:
    """
    Process image seam carving for removal and reconstruction.
    """

    def __init__(self, image_file: str) -> None:
        image_folder = "data/benchmark/"
        self.image_path = image_folder + image_file
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load the image from path: {self.image_path}")
        self.height, self.width = self.image.shape[:2]
        self.original_image = self.image.copy()  # Save a copy of the original image

    def calculate_energy_map(self) -> np.ndarray:
        """
        Calculate the energy map of the image using the gradient magnitude.
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy_map = np.abs(grad_x) + np.abs(grad_y)
        return energy_map

    def find_seam(self, energy_map: np.ndarray) -> np.ndarray:
        """
        Find the seam with the minimum energy using dynamic programming.
        """
        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=int)  # Use 'int' instead of 'np.int'

        for i in range(1, self.height):
            for j in range(self.width):
                if j == 0:
                    idx = np.argmin(M[i - 1, j : j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1 : j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]
                M[i, j] += min_energy

        return backtrack

    def remove_seam(self, seam: np.ndarray) -> None:
        """
        Remove the seam from the image.
        """
        output = np.zeros((self.height, self.width - 1, 3), dtype=np.uint8)
        for i in range(self.height):
            j = int(seam[i, 0])  # Ensure to extract the integer index properly
            # Copy the left part of the seam
            output[i, :j, :] = self.image[i, :j, :]
            # Copy the right part of the seam
            output[i, j:, :] = self.image[i, j + 1 :, :]
        self.image = output
        self.width -= 1

    def calculate_mse(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE) between two images.
        """
        mse = np.mean((image1 - image2) ** 2)
        return mse

    def save_output(self, filename: str) -> None:
        """
        Save the carved image to the specified path.
        """
        base_path = "data/seamcarving/"
        full_path = os.path.join(base_path, filename)
        cv2.imwrite(full_path, self.image)
        print(f"Saved: {full_path}")

    def process_seam_carving(self) -> None:
        """
        Process the seam carving of the image and save the results of the image with minimum MSE.
        """
        num_seams = 50
        min_mse = float("inf")  # Initialize with a very large number
        best_image = None
        best_output_filename = ""

        mse_values = []  # List to store all MSE values

        for _ in range(num_seams):
            energy_map = self.calculate_energy_map()
            seam = self.find_seam(energy_map)
            self.remove_seam(seam)

            # Resize original_image to match current image dimensions
            self.original_image = cv2.resize(
                self.original_image, (self.width, self.height)
            )

            # Calculate MSE between original image and current image
            current_image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            original_image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            mse = self.calculate_mse(original_image_gray, current_image_gray)

            mse_values.append(mse)  # Append current MSE to the list

            # Check if current MSE is lower than the minimum MSE found so far
            if mse < min_mse:
                min_mse = mse
                best_image = self.image.copy()  # Make a copy of the current best image
                # Construct the output filename with the original image file name
                best_output_filename = f"{os.path.splitext(os.path.basename(self.image_path))[0]}-best-{_ + 1}-seams-carved.jpg"

        # Print all MSE values
        print("All MSE values:")
        for i, mse in enumerate(mse_values):
            print(f"Iteration {_ + 1}: {mse}")

        # Save the image with the lowest MSE
        if best_image is not None:
            self.image = best_image
            self.save_output(best_output_filename)
            print(f"Minimum MSE image saved as: {best_output_filename}")
