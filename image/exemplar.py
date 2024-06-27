import os
import cv2
import numpy as np
from tqdm import tqdm


# Reference: Tai, Y. W., Jia, J., & Tang, C. K. (2009). Non-parametric patch-based image inpainting. IEEE Transactions on Image Processing, 18(1), 27-35. https://doi.org/10.1109/TIP.2008.2008281
class ExemplarBasedInpainter:
    """
    Performing exemplar-based inpainting on images using masks.
    """

    def split_filename(self, file_name):
        """
        Split the filename into the name without the extension and the extension.
        """
        file_name_without_extension, file_extension = os.path.splitext(file_name)
        return file_name_without_extension, file_extension

    def find_removable_regions(self, image):
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

    def save_mask_to_file(self, mask, filename):
        """
        Save the mask to a text file.
        """
        np.savetxt(filename, mask, fmt="%d")
        print(f"Saved mask to {filename}")

    def load_mask_from_file(self, filename):
        """
        Load the mask from a text file.
        """
        return np.loadtxt(filename, dtype=np.uint8)

    def inpaint_image(self, image_path, mask_filename):
        """
        Perform exemplar-based inpainting on the image using the provided mask.
        """
        # Load image
        image = cv2.imread(image_path)

        # Find removable regions
        mask = self.find_removable_regions(image)

        # Save the mask to a text file
        self.save_mask_to_file(mask, mask_filename)

        # Perform exemplar-based inpainting
        exemplar_inpainting_image = self.exemplar_inpaint(image, mask)

        return exemplar_inpainting_image

    def exemplar_inpaint(self, image, mask):
        """
        Perform exemplar-based inpainting using patches from the image.
        """
        # Ensure the mask is single-channel and binary
        mask = (mask > 0).astype(np.uint8)

        # Perform inpainting
        inpainted_image = image.copy()

        # Iterate over each pixel in the mask with tqdm for progress tracking
        total_pixels = np.count_nonzero(mask)
        progress_bar = tqdm(
            total=total_pixels, desc="Inpainting Progress", unit="pixels"
        )

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if mask[y, x] > 0:
                    # Select an exemplar patch from a similar region in the image
                    exemplar_patch = self.select_exemplar_patch(image, mask, (x, y))

                    # Replace the masked region with the exemplar patch
                    inpainted_image[y, x] = exemplar_patch[
                        0, 0
                    ]  # Example: Assigning the top-left pixel value

                    progress_bar.update(1)  # Update progress bar

        progress_bar.close()  # Close the progress bar after completion

        return inpainted_image

    def select_exemplar_patch(self, image, mask, center):
        """
        Select an exemplar patch from the image based on similarity to the region around 'center'.
        """
        patch_size = 9  # Size of the patch to search around the center point
        half_size = patch_size // 2

        # Get coordinates of the center
        cx, cy = center

        # Ensure the patch stays within image boundaries
        h, w = image.shape[:2]
        min_x = max(cx - half_size, 0)
        max_x = min(cx + half_size + 1, w)
        min_y = max(cy - half_size, 0)
        max_y = min(cy + half_size + 1, h)

        # Extract the target region to inpaint (considering the patch size)
        target_region = image[min_y:max_y, min_x:max_x].astype(np.float32)
        target_mask = mask[min_y:max_y, min_x:max_x]

        # Compute SSD (Sum of Squared Differences) or NCC (Normalized Cross-Correlation)
        best_similarity = -1
        best_patch = None

        for y in range(h - patch_size + 1):
            for x in range(w - patch_size + 1):
                # Ensure candidate patch stays within image boundaries
                if x + patch_size > w or y + patch_size > h:
                    continue

                # Extract patch from image to compare
                candidate_patch = image[y : y + patch_size, x : x + patch_size].astype(
                    np.float32
                )

                # Ensure both patches are of the same size
                if candidate_patch.shape != target_region.shape:
                    continue

                # Compute similarity metric (SSD or NCC)
                similarity = self.compute_similarity(
                    target_region, candidate_patch, target_mask
                )

                # Update best patch if found a better match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_patch = candidate_patch

        # Check if a valid patch was found
        if best_patch is not None:
            return best_patch.astype(np.uint8)
        else:
            return np.zeros_like(
                target_region, dtype=np.uint8
            )  # Return a blank patch or handle this case as needed

    def compute_similarity(self, target_region, candidate_patch, target_mask):
        """
        Compute similarity between target region and candidate patch using SSD (Sum of Squared Differences).
        """
        # Calculate SSD (Sum of Squared Differences)
        diff = target_region - candidate_patch
        ssd = np.sum(diff**2)

        # Normalize SSD by the number of pixels in the region
        similarity = 1.0 - (
            ssd / (np.count_nonzero(target_mask) * 255.0 * target_region.size)
        )

        return similarity

    def save_output(self, image, filename):
        """
        Save the output image to a file.
        """
        base_path = "data/exemplar/"
        full_path = os.path.join(base_path, filename)
        cv2.imwrite(full_path, image)
        print(f"Saved: {full_path}")

    def compute_mse(self, image1, image2):
        """
        Compute Mean Squared Error (MSE) between two images.
        """
        mse = np.mean((image1 - image2) ** 2)
        return mse

    def process_inpaint(self, file_name):
        """
        Process the exemplar-based inpainting of the image and save the results.
        Print MSE between original and inpainted image.
        """
        file_name_without_extension, _ = self.split_filename(file_name)
        image_path = f"data/benchmark/{file_name}"
        mask_filename = f"data/exemplar/{file_name_without_extension}-mask.txt"

        # Load original image
        original_image = cv2.imread(image_path)

        # Perform exemplar-based inpainting
        output_exemplar = self.inpaint_image(image_path, mask_filename)

        # Save the inpainted image
        self.save_output(
            output_exemplar, f"{file_name_without_extension}_inpainted_exemplar.jpg"
        )

        # Compute MSE between original and inpainted image
        mse = self.compute_mse(original_image, output_exemplar)
        print(f"MSE between original and inpainted image: {mse}")
