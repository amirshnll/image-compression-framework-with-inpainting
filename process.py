import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from image import (
    ImageProcessing,
    IQA,
    Inpainting,
    Channel,
)
from utils import ClearProject, TypeCaster, ProcessLogger, FileHandler, get_object_size
import pandas as pd


class Main:
    def __init__(self, input_folder: str = "data/benchmark") -> None:
        self.input_folder = input_folder

    def create_directory(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def clear(self) -> None:
        clear_project = ClearProject()
        clear_project.run()

    def process_single_image(
        self,
        image_file: str,
        quality_values: list,
        threshold_values: list,
        edge_methods: list,
        encoding_methods: list,
        inpaint_methods: list,
        inpaint_radii: list,
    ) -> None:
        file_path = os.path.join(self.input_folder, image_file)
        image = ImageProcessing().open(file_path)
        original_size = os.path.getsize(file_path)
        inpainting = Inpainting(image)
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()
        results = []

        for quality in tqdm(quality_values, desc="Quality"):
            for threshold in tqdm(threshold_values, desc="Threshold", leave=False):
                for edge_method in tqdm(edge_methods, desc="Edge Method", leave=False):
                    for encoding_method in tqdm(
                        encoding_methods, desc="Encoding Method", leave=False
                    ):
                        for inpaint_method in tqdm(
                            inpaint_methods, desc="Inpaint Method", leave=False
                        ):
                            for inpaint_radius in tqdm(
                                inpaint_radii, desc="Inpaint Radius", leave=False
                            ):
                                logger.start_timer()
                                metadata = inpainting.custom_compression(
                                    quality=quality,
                                    threshold=threshold,
                                    method="high_intensity_and_edge",
                                )
                                metadata_str = caster.dict_to_str(
                                    {
                                        "original_shape": caster.tuple_to_str(
                                            metadata["original_shape"]
                                        ),
                                        "inpainted_shape": caster.tuple_to_str(
                                            metadata["inpainted_shape"]
                                        ),
                                        "compressed_size": caster.int_to_str(
                                            metadata["compressed_size"]
                                        ),
                                        "mask_shape": caster.tuple_to_str(
                                            metadata["mask_shape"]
                                        ),
                                        "mask_non_zero_count": caster.int_to_str(
                                            metadata["mask_non_zero_count"]
                                        ),
                                        "compressed_data": caster.ndarray_to_str(
                                            metadata["compressed_data"]
                                        ),
                                    }
                                )
                                encoded_data = channel.encode_and_transmit(
                                    metadata_str, encoding_method
                                )
                                output_file = f"data/channel/{Path(image_file).stem}-{encoding_method}-edge_{edge_method}-quality{quality}-threshold{threshold}-inpaint_{inpaint_method}-radius{inpaint_radius}.txt"

                                file_handler = FileHandler(output_file)
                                file_handler.write_file(str(encoded_data))
                                encoded_size = os.path.getsize(output_file)
                                logger.end_timer()
                                processing_time = logger.processing_time

                                # Restore Process
                                logger.start_timer()
                                received_data = file_handler.read_file()
                                decoded_metadata_str = channel.receive_and_decode(
                                    received_data, encoding_method
                                )
                                decoded_metadata = caster.str_to_dict(
                                    decoded_metadata_str
                                )
                                decoded_metadata = {
                                    "original_shape": caster.str_to_tuple(
                                        decoded_metadata["original_shape"]
                                    ),
                                    "inpainted_shape": caster.str_to_tuple(
                                        decoded_metadata["inpainted_shape"]
                                    ),
                                    "compressed_size": caster.str_to_int(
                                        decoded_metadata["compressed_size"]
                                    ),
                                    "mask_shape": caster.str_to_tuple(
                                        decoded_metadata["mask_shape"]
                                    ),
                                    "mask_non_zero_count": caster.str_to_int(
                                        decoded_metadata["mask_non_zero_count"]
                                    ),
                                    "compressed_data": caster.str_to_ndarray(
                                        decoded_metadata["compressed_data"]
                                    ),
                                }
                                restored_image = inpainting.restore_image(
                                    decoded_metadata,
                                    inpaint_method=inpaint_method,
                                    inpaint_radius=inpaint_radius,
                                )
                                logger.end_timer()

                                output_folder = "data/restored"
                                self.create_directory(output_folder)
                                output_image_file = os.path.join(
                                    output_folder,
                                    f"{Path(image_file).stem}-{encoding_method}-edge_{edge_method}-quality{quality}-threshold{threshold}-inpaint_{inpaint_method}-radius{inpaint_radius}.png",
                                )
                                ImageProcessing().save_image(
                                    restored_image, output_image_file
                                )

                                # IQA Comparison
                                iqa = IQA(np.array(image), restored_image)
                                mse_value = iqa.calculate_mse()
                                psnr_value = iqa.calculate_psnr(mse_value)
                                ssim_value = iqa.calculate_ssim()
                                rmse_value = iqa.calculate_rmse()
                                mae_value = iqa.calculate_mae()
                                entropy_values = iqa.calculate_entropy()
                                compression_ratio = iqa.calculate_compression_ratio(
                                    restored_image
                                )
                                bitrate = iqa.calculate_bitrate(restored_image)
                                fsim_value = iqa.calculate_fsim()

                                # Calculate combined metric
                                combined_metric = (
                                    0.2 * psnr_value
                                    + 0.2 * ssim_value
                                    + 0.2 * compression_ratio
                                    - 0.2 * rmse_value
                                    - 0.2 * mse_value
                                )

                                results.append(
                                    {
                                        "image_file": image_file,
                                        "encoding_method": encoding_method,
                                        "edge_method": edge_method,
                                        "quality": quality,
                                        "threshold": threshold,
                                        "inpaint_method": inpaint_method,
                                        "inpaint_radius": inpaint_radius,
                                        "encoded_size": encoded_size,
                                        "percentage_decrease": (
                                            (original_size - encoded_size)
                                            / original_size
                                        )
                                        * 100,
                                        "processing_time": processing_time,
                                        "mse": mse_value,
                                        "psnr": psnr_value,
                                        "ssim": ssim_value,
                                        "rmse": rmse_value,
                                        "mae": mae_value,
                                        "entropy": entropy_values,
                                        "compression_ratio": compression_ratio,
                                        "bitrate": bitrate,
                                        "fsim": fsim_value,
                                        "combined_metric": combined_metric,
                                    }
                                )

        # Convert results to DataFrame for easy sorting and filtering
        df_results = pd.DataFrame(results)

        # Sorting and selecting top and bottom 10 for each metric
        top_10_encoded_size = df_results.nsmallest(10, "encoded_size")
        bottom_10_encoded_size = df_results.nlargest(10, "encoded_size")

        top_10_percentage_decrease = df_results.nlargest(10, "percentage_decrease")
        bottom_10_percentage_decrease = df_results.nsmallest(10, "percentage_decrease")

        top_10_processing_time = df_results.nsmallest(10, "processing_time")
        bottom_10_processing_time = df_results.nlargest(10, "processing_time")

        top_10_mse = df_results.nsmallest(10, "mse")
        bottom_10_mse = df_results.nlargest(10, "mse")

        top_10_combined = df_results.nlargest(10, "combined_metric")
        bottom_10_combined = df_results.nsmallest(10, "combined_metric")

        # Display the results
        print("Top 10 by Encoded Size:")
        print(
            top_10_encoded_size[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "encoded_size",
                ]
            ]
        )

        print("\nBottom 10 by Encoded Size:")
        print(
            bottom_10_encoded_size[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "encoded_size",
                ]
            ]
        )

        print("\nTop 10 by Percentage Decrease:")
        print(
            top_10_percentage_decrease[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "percentage_decrease",
                ]
            ]
        )

        print("\nBottom 10 by Percentage Decrease:")
        print(
            bottom_10_percentage_decrease[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "percentage_decrease",
                ]
            ]
        )

        print("\nTop 10 by Processing Time:")
        print(
            top_10_processing_time[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "processing_time",
                ]
            ]
        )

        print("\nBottom 10 by Processing Time:")
        print(
            bottom_10_processing_time[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "processing_time",
                ]
            ]
        )

        print("\nTop 10 by MSE:")
        print(
            top_10_mse[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "mse",
                ]
            ]
        )

        print("\nBottom 10 by MSE:")
        print(
            bottom_10_mse[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "mse",
                ]
            ]
        )

        print("\nTop 10 by Combined Metric:")
        print(
            top_10_combined[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "combined_metric",
                ]
            ]
        )

        print("\nBottom 10 by Combined Metric:")
        print(
            bottom_10_combined[
                [
                    "image_file",
                    "encoding_method",
                    "edge_method",
                    "quality",
                    "threshold",
                    "inpaint_method",
                    "inpaint_radius",
                    "combined_metric",
                ]
            ]
        )


if __name__ == "__main__":
    main = Main()
    main.clear()

    # All variables
    quality_values = range(0, 101, 10)
    threshold_values = range(0, 256, 25)
    inpaint_methods = ["telea", "ns"]
    inpaint_radii = range(3, 23, 2)
    edge_methods = [
        "sobel",
        "prewitt",
        "roberts",
        "log",
        "canny",
        "scharr",
        "zero_crossing",
        "optimal_canny",
    ]
    encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]

    main.process_single_image(
        image_file="lena.png",
        quality_values=quality_values,
        threshold_values=threshold_values,
        edge_methods=edge_methods,
        encoding_methods=encoding_methods,
        inpaint_methods=inpaint_methods,
        inpaint_radii=inpaint_radii,
    )
