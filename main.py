import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from image import (
    ImageProcessing,
    EdgeDetection,
    Plot,
    ImageCompression,
    IQA,
    Inpainting,
    Channel,
    Encoding,
)
from utils import ClearProject, TypeCaster, ProcessLogger, FileHandler, get_object_size
import pandas as pd


class Main:
    def __init__(self, input_folder: str = "data/benchmark") -> None:
        """
        Initializes the class instance.
        """
        self.input_folder = input_folder

    def create_directory(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_image_files(self, folder: str, extensions=(".png", ".jpg", ".jpeg")):
        return [f for f in os.listdir(folder) if f.endswith(extensions)]

    def get_encoded_files(self, folder: str, pattern: str):
        return glob.glob(os.path.join(folder, pattern))

    def get_base_name(self, file_path: str):
        return os.path.splitext(os.path.basename(file_path))[0]

    def get_file_size(self, file_path: str):
        return os.path.getsize(file_path)

    def save_image(self, image, path: str):
        ImageProcessing().save_image(image, path)

    def save_image(self, image, path: str):
        ImageProcessing().save_image(image, path)

    def start_process_logger(self):
        logger = ProcessLogger()
        logger.start_timer()
        return logger

    def end_process_logger(self, logger):
        logger.end_timer()
        return logger.processing_time

    def encode_metadata(self, metadata, caster, channel, encoding_method):
        metadata_str = caster.dict_to_str(
            {
                "original_shape": caster.tuple_to_str(metadata["original_shape"]),
                "inpainted_shape": caster.tuple_to_str(metadata["inpainted_shape"]),
                "compressed_size": caster.int_to_str(metadata["compressed_size"]),
                "mask_shape": caster.tuple_to_str(metadata["mask_shape"]),
                "mask_non_zero_count": caster.int_to_str(
                    metadata["mask_non_zero_count"]
                ),
                "compressed_data": caster.ndarray_to_str(metadata["compressed_data"]),
            }
        )
        return channel.encode_and_transmit(metadata_str, encoding_method)

    def decode_metadata(self, encoded_data, caster, channel, encoding_method):
        decoded_metadata_str = channel.receive_and_decode(encoded_data, encoding_method)
        decoded_metadata = caster.str_to_dict(decoded_metadata_str)
        return {
            "original_shape": caster.str_to_tuple(decoded_metadata["original_shape"]),
            "inpainted_shape": caster.str_to_tuple(decoded_metadata["inpainted_shape"]),
            "compressed_size": caster.str_to_int(decoded_metadata["compressed_size"]),
            "mask_shape": caster.str_to_tuple(decoded_metadata["mask_shape"]),
            "mask_non_zero_count": caster.str_to_int(
                decoded_metadata["mask_non_zero_count"]
            ),
            "compressed_data": caster.str_to_ndarray(
                decoded_metadata["compressed_data"]
            ),
        }

    def log_process(
        self, logger, original_size, encoded_size, image_file, encoding_method
    ):
        logger.log_process(original_size, encoded_size, image_file, encoding_method)

    def iqa_comparison(self, original_image, restored_image):
        iqa = IQA(np.array(original_image), restored_image)
        return {
            "mse": iqa.calculate_mse(),
            "psnr": iqa.calculate_psnr(),
            "ssim": iqa.calculate_ssim(),
            "rmse": iqa.calculate_rmse(),
            "mae": iqa.calculate_mae(),
            "entropy": iqa.calculate_entropy(),
            "compression_ratio": iqa.calculate_compression_ratio(restored_image),
            "bitrate": iqa.calculate_bitrate(restored_image),
            "fsim": iqa.calculate_fsim(),
        }

    def display_results(self, results, top_n=10):
        df_results = pd.DataFrame(results)

        top_10_percentage_decrease = df_results.nlargest(top_n, "percentage_decrease")
        bottom_10_percentage_decrease = df_results.nsmallest(
            top_n, "percentage_decrease"
        )

        top_10_processing_time = df_results.nsmallest(top_n, "processing_time")
        bottom_10_processing_time = df_results.nlargest(top_n, "processing_time")

        top_10_mse = df_results.nsmallest(top_n, "mse")
        bottom_10_mse = df_results.nlargest(top_n, "mse")

        top_10_combined = df_results.nlargest(top_n, "combined_metric")
        bottom_10_combined = df_results.nsmallest(top_n, "combined_metric")

        print("Top 10 by Percentage Decrease:")
        print(
            top_10_percentage_decrease[
                [
                    "image_file",
                    "encoding_method",
                    "quality",
                    "threshold",
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
                    "quality",
                    "threshold",
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
                    "quality",
                    "threshold",
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
                    "quality",
                    "threshold",
                    "processing_time",
                ]
            ]
        )

        print("\nTop 10 by MSE:")
        print(
            top_10_mse[["image_file", "encoding_method", "quality", "threshold", "mse"]]
        )

        print("\nBottom 10 by MSE:")
        print(
            bottom_10_mse[
                ["image_file", "encoding_method", "quality", "threshold", "mse"]
            ]
        )

        print("\nTop 10 by Combined Metric:")
        print(
            top_10_combined[
                [
                    "image_file",
                    "encoding_method",
                    "quality",
                    "threshold",
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
                    "quality",
                    "threshold",
                    "combined_metric",
                ]
            ]
        )

    def clear(self) -> None:
        """
        Clear the project workspace
        """
        clear_project = ClearProject()
        clear_project.run()

    def edge_detection(self) -> None:
        """
        Perform edge detection on benchmark images
        """
        image_folder = "data/benchmark"
        plotter = Plot()
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
        image_files = self.get_image_files(image_folder)
        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(image_folder, image_file)
            image = ImageProcessing().open(file_path)
            edge_detector = EdgeDetection(image)
            results = []
            for method in edge_methods:
                edge_func = getattr(edge_detector, method)
                edge_image = edge_func()
                results.append({"title": method, "image": edge_image})
            plot_title = f"Edge Detection Results for {image_file}"
            # plotter.show_plot(
            #     data=results, plot_type="image", title=plot_title, subplot_layout=(2, 4)
            # )
            plotter.save_plot(data=results, title=plot_title, subplot_layout=(2, 4))

    def jpeg_compression(self) -> None:
        """
        Apply JPEG compression to images
        """
        image_files = self.get_image_files(self.input_folder)

        logger = self.start_process_logger()

        for image_file in tqdm(image_files, desc="Compressing images"):
            file_path = os.path.join(self.input_folder, image_file)
            img = ImageProcessing()
            image = img.open(file_path)
            compressor = ImageCompression(image)

            logger.start_timer()
            compressed_image = compressor.compress_jpeg()
            decompressed_image = compressor.decompress_jpeg(compressed_image)
            logger.end_timer()

            compressed_image = compressor.decompress_and_decode(compressed_image)

            plotter = Plot()
            # plotter.show_plot(
            #     data=[
            #         {
            #             "title": "Original Image",
            #             "image": img.convert_to_rgb(image),
            #         },
            #         {
            #             "title": "Compressed Image",
            #             "image": img.convert_to_rgb(compressed_image),
            #         },
            #         {
            #             "title": "Decompressed Image",
            #             "image": img.convert_to_rgb(decompressed_image),
            #         },
            #     ]
            # )

            plotter.save_plot(
                data=[
                    {
                        "title": "Original Image",
                        "image": img.convert_to_rgb(image),
                    },
                    {
                        "title": "Compressed Image",
                        "image": img.convert_to_rgb(compressed_image),
                    },
                    {
                        "title": "Decompressed Image",
                        "image": img.convert_to_rgb(decompressed_image),
                    },
                ],
            )

            print(
                f"Processing time for {image_file}: {logger.processing_time:.2f} seconds"
            )

    def image_compression(self) -> None:
        """
        Compress images using various compression methods and store the result
        """
        output_folder = "data/compressed"
        self.create_directory(output_folder)
        compression_methods = [
            "compress_jpeg",
            "compress_png",
            "compress_webp",
            "compress_tiff",
            "compress_jpeg2000",
            "compress_avif",
        ]
        file_extensions = {
            "compress_jpeg": ".jpg",
            "compress_png": ".png",
            "compress_webp": ".webp",
            "compress_tiff": ".tiff",
            "compress_jpeg2000": ".jp2",
            "compress_avif": ".avif",
        }
        image_files = self.get_image_files(self.input_folder)
        for image_file in tqdm(image_files, desc="Compressing images"):
            file_path = os.path.join(self.input_folder, image_file)
            img = ImageProcessing().open(file_path)
            image = np.array(img)
            compressor = ImageCompression(image)

            for method in compression_methods:
                compression_func = getattr(compressor, method)
                compressed_image = compression_func()
                base_name = self.get_base_name(image_file)
                ext = file_extensions[method]
                output_file_path = os.path.join(
                    output_folder, f"{base_name}-{method}{ext}"
                )
                file_handler = FileHandler(output_file_path)
                file_handler.write_binary_file(compressed_image)

    def iqa_original_by_compressed(self) -> None:
        """
        Conduct image quality assessment (IQA) by comparing original and compressed images
        """
        image_files = self.get_image_files(self.input_folder)
        img = ImageProcessing()

        for image_file in image_files:
            original_image_path = os.path.join(self.input_folder, image_file)
            original_image = np.array(img.open(original_image_path))

            compressed_folder: str = "data/compressed"

            base_name = self.get_base_name(image_file)
            compressed_files = [
                f
                for f in os.listdir(compressed_folder)
                if f.startswith(base_name) and not f == image_file
            ]

            for compressed_file in compressed_files:
                compressed_image_path = os.path.join(compressed_folder, compressed_file)
                compressed_image = img.open(compressed_image_path)

                iqa = IQA(original_image, compressed_image)
                mse_value = iqa.calculate_mse()
                psnr_value = iqa.calculate_psnr(mse_value)
                ssim_value = iqa.calculate_ssim()
                rmse_value = iqa.calculate_rmse()
                mae_value = iqa.calculate_mae()
                entropy_values = iqa.calculate_entropy()
                compression_ratio = iqa.calculate_compression_ratio(compressed_image)
                bitrate = iqa.calculate_bitrate(compressed_image)
                fsim_value = iqa.calculate_fsim()

                print(f"Original Image: {original_image_path}")
                print(f"Compressed Image: {compressed_image_path}")
                print(f"MSE: {mse_value}")
                print(f"PSNR: {psnr_value}")
                print(f"SSIM: {ssim_value}")
                print(f"RMSE: {rmse_value}")
                print(f"MAE: {mae_value}")
                print(f"Entropy: {entropy_values}")
                print(f"Compression Ratio: {compression_ratio}")
                print(f"Bitrate: {bitrate}")
                print(f"FSIM: {fsim_value}")
                print("")

    def process_mask_images(self) -> None:
        """
        Process images to create masks using different parameters
        """
        output_folder = "data/mask"
        self.create_directory(output_folder)
        image_files = self.get_image_files(self.input_folder)
        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            img = ImageProcessing()
            original_image = img.open(file_path)

            inpainting_processor = Inpainting(original_image)
            mask_high_intensity = (
                inpainting_processor.select_removable_area_by_high_intensity()
            )
            mask_high_intensity_path = os.path.join(
                output_folder, f"{self.get_base_name(image_file)}-high_intensity.png"
            )
            self.save_image(mask_high_intensity, mask_high_intensity_path)

            edge_detector = EdgeDetection(original_image)
            mask_high_intensity_and_edge_canny = (
                inpainting_processor.select_removable_area_by_high_intensity_and_edge()
            )
            mask_high_intensity_and_edge_path_canny = os.path.join(
                output_folder,
                f"{self.get_base_name(image_file)}-high_intensity_edge_canny.png",
            )
            self.save_image(
                mask_high_intensity_and_edge_canny,
                mask_high_intensity_and_edge_path_canny,
            )

            mask_high_intensity_and_edge_optimal_canny = (
                inpainting_processor.select_removable_area_by_high_intensity_and_edge(
                    edge_method="optimal_canny"
                )
            )
            mask_high_intensity_and_edge_path_optimal_canny = os.path.join(
                output_folder,
                f"{self.get_base_name(image_file)}-high_intensity_edge_canny.png",
            )
            self.save_image(
                mask_high_intensity_and_edge_optimal_canny,
                mask_high_intensity_and_edge_path_optimal_canny,
            )

    def inpainting(self) -> None:
        """
        Perform inpainting on images and inpainting
        """
        image_files = self.get_image_files(self.input_folder)
        logger = ProcessLogger()

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            inpainting = Inpainting(image)

            logger.start_timer()
            metadata = inpainting.custom_compression(
                quality=80, threshold=50, method="high_intensity"
            )
            logger.end_timer()

            original_size = self.get_file_size(file_path)
            metadata_size = get_object_size(metadata["compressed_data"])

            print(
                f"Original size: {original_size} bytes, Metadata size: {metadata_size} bytes, "
                f"Percentage distance: {((original_size - metadata_size) / original_size) * 100:.2f}%, "
                f"Processing time: {logger.processing_time:.2f} seconds"
            )

    def inpainting_by_edge(self) -> None:
        """
        Perform inpainting on images and inpainting using edge detection
        """
        image_files = self.get_image_files(self.input_folder)
        logger = ProcessLogger()

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            inpainting = Inpainting(image)

            logger.start_timer()
            metadata = inpainting.custom_compression(
                quality=80, threshold=50, method="high_intensity_and_edge"
            )
            logger.end_timer()

            original_size = self.get_file_size(file_path)
            metadata_size = get_object_size(metadata["compressed_data"])

            print(
                f"Original size: {original_size} bytes, Metadata size: {metadata_size} bytes, "
                f"Percentage distance: {((original_size - metadata_size) / original_size) * 100:.2f}%, "
                f"Processing time: {logger.processing_time:.2f} seconds"
            )

    def encoding_algorithms(self, input_text: str) -> None:
        """
        Apply various encoding algorithms to a text string and verify correctness
        """
        encoder = Encoding()

        methods = ["rle", "hamming", "huffman", "base64", "ascii"]

        for method in methods:
            encoded = encoder.encode(input_text, method)
            decoded = encoder.decode(encoded, method)
            print("")
            print(method)
            print("encoded", encoded)
            print("decoded", decoded)
            print("is_correct", decoded == input_text)
            print("")

    def encoding_algorithms_by_channel(self, input_text: str) -> None:
        """
        Apply various encoding algorithms to a text string and verify correctness
        """
        channel = Channel()

        methods = ["rle", "hamming", "huffman", "base64", "ascii"]

        for method in methods:
            encoded = channel.encode_and_transmit(input_text, method)
            decoded = channel.receive_and_decode(encoded, method)
            print("")
            print(method)
            print("encoded", encoded)
            print("decoded", decoded)
            print("is_correct", decoded == input_text)
            print("")

    def verify_image_conversion(self) -> None:
        """
        Verify image conversion by converting to string and back to ndarray
        """
        image_files = self.get_image_files(self.input_folder)
        caster = TypeCaster()
        for image_file in tqdm(image_files, desc="Processing images"):

            original_image_path = os.path.join(self.input_folder, "lena.png")
            original_image_array = ImageProcessing().open(original_image_path)

            encoded_str = caster.ndarray_to_str(original_image_array)
            restored_array = caster.str_to_ndarray(encoded_str)
            are_equal = np.array_equal(original_image_array, restored_array)
            print("Arrays are equal:", are_equal)

    def process_images_without_inpainting(self) -> None:
        """
        Process images without inpainting
        """
        image_files = self.get_image_files(self.input_folder)

        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)

            original_size = self.get_file_size(file_path)

            metadata = caster.ndarray_to_str(image)

            for encoding_method in encoding_methods:
                logger.start_timer()
                encoded_data = channel.encode_and_transmit(metadata, encoding_method)
                logger.end_timer()

                output_file = f"data/channel/{Path(image_file).stem}-without-inpainting-{encoding_method}.txt"

                file_handler = FileHandler(output_file)
                file_handler.write_file(str(encoded_data))

                encoded_size = self.get_file_size(output_file)

                self.log_process(
                    logger, original_size, encoded_size, image_file, encoding_method
                )

    def restore_process_images_without_inpainting(self) -> None:
        """
        Restore images processed without inpainting
        """
        encoded_files = self.get_encoded_files(
            "data/channel", "*-without-inpainting-*.txt"
        )
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        output_folder = "data/restored"
        self.create_directory(output_folder)

        for encoded_file in tqdm(encoded_files, desc="Restoring files"):
            encoding_method = Path(encoded_file).stem.split("-")[-1]
            image_name = Path(encoded_file).stem.replace(
                f"-without-inpainting-{encoding_method}", ""
            )

            original_image_path = os.path.join(self.input_folder, f"{image_name}.png")
            if not os.path.exists(original_image_path):
                original_image_path = os.path.join(
                    self.input_folder, f"{image_name}.jpg"
                )
            if not os.path.exists(original_image_path):
                original_image_path = os.path.join(
                    self.input_folder, f"{image_name}.jpeg"
                )

            if not os.path.exists(original_image_path):
                print(f"Original image for {encoded_file} not found.")
                continue

            file_handler = FileHandler(encoded_file)
            encoded_data = file_handler.read_file()

            logger.start_timer()
            metadata = channel.receive_and_decode(encoded_data, encoding_method)
            metadata = caster.str_to_ndarray(metadata)
            restored_image = metadata
            logger.end_timer()

            original_image = ImageProcessing().open(original_image_path)

            output_image_file = os.path.join(
                output_folder, f"{Path(encoded_file).stem}.png"
            )
            self.save_image(restored_image, output_image_file)

            iqa = IQA(original_image, restored_image)
            metrics = self.iqa_comparison(original_image, restored_image)

            original_size = self.get_file_size(original_image_path)
            encoded_size = self.get_file_size(encoded_file)

            self.log_process(
                logger, original_size, encoded_size, image_name, encoding_method
            )

            print(f"Restored {encoded_file} and saved to {output_image_file}")
            print(f"IQA Results for {encoded_file}:")
            for metric, value in metrics.items():
                print(f"{metric.upper()}: {value}")
            print(f"Processing time: {logger.processing_time:.2f} seconds\n")

    def process_images(self) -> None:
        """
        Process images with inpainting and transmit through channel encoding
        """
        image_files = self.get_image_files(self.input_folder)
        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            original_size = self.get_file_size(file_path)
            inpainting = Inpainting(image)

            for encoding_method in encoding_methods:
                logger.start_timer()
                metadata = inpainting.custom_compression()

                metadata = caster.dict_to_str(
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
                        "mask_shape": caster.tuple_to_str(metadata["mask_shape"]),
                        "mask_non_zero_count": caster.int_to_str(
                            metadata["mask_non_zero_count"]
                        ),
                        "compressed_data": caster.ndarray_to_str(
                            metadata["compressed_data"]
                        ),
                    }
                )

                encoded_data = channel.encode_and_transmit(metadata, encoding_method)

                output_file = (
                    f"data/channel/{Path(image_file).stem}-{encoding_method}.txt"
                )

                file_handler = FileHandler(output_file)
                file_handler.write_file(str(encoded_data))

                encoded_size = self.get_file_size(output_file)

                logger.end_timer()
                self.log_process(
                    logger, original_size, encoded_size, image_file, encoding_method
                )

    def restore_process_images(self) -> None:
        """
        Restore images processed with inpainting and transmitted through channel encoding
        """
        encoded_files = self.get_encoded_files("data/channel", "*.txt")
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        for encoded_file in tqdm(encoded_files, desc="Processing files"):
            encoding_method = Path(encoded_file).stem.split("-")[-1]

            file_handler = FileHandler(encoded_file)
            encoded_data = file_handler.read_file()

            logger.start_timer()
            metadata = channel.receive_and_decode(encoded_data, encoding_method)

            metadata = caster.str_to_dict(metadata)
            metadata = {
                "original_shape": caster.str_to_tuple(metadata["original_shape"]),
                "inpainted_shape": caster.str_to_tuple(metadata["inpainted_shape"]),
                "compressed_size": caster.str_to_int(metadata["compressed_size"]),
                "mask_shape": caster.str_to_tuple(metadata["mask_shape"]),
                "mask_non_zero_count": caster.str_to_int(
                    metadata["mask_non_zero_count"]
                ),
                "compressed_data": caster.str_to_ndarray(metadata["compressed_data"]),
            }

            inpainting = Inpainting()
            restored_image = inpainting.restore_image(metadata)
            logger.end_timer()

            output_image_file = f"data/restored/{Path(encoded_file).stem}.png"
            self.save_image(restored_image, output_image_file)

            print(
                f"Processing time for {output_image_file}: {logger.processing_time:.2f} seconds"
            )

    def process_and_restore_inpainting_by_iqa(self) -> None:
        """
        Process and restore inpainting images with image quality assessment (IQA)
        """
        image_files = self.get_image_files(self.input_folder)
        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        output_folder = "data/restored"
        self.create_directory(output_folder)

        for image_file in tqdm(image_files, desc="Processing and restoring images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            original_size = self.get_file_size(file_path)
            inpainting = Inpainting(image)

            for encoding_method in encoding_methods:
                logger.start_timer()
                metadata = inpainting.custom_compression()
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
                        "mask_shape": caster.tuple_to_str(metadata["mask_shape"]),
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
                output_file = (
                    f"data/channel/{Path(image_file).stem}-{encoding_method}.txt"
                )
                file_handler = FileHandler(output_file)
                file_handler.write_file(str(encoded_data))
                encoded_size = self.get_file_size(output_file)
                logger.end_timer()
                self.log_process(
                    logger, original_size, encoded_size, image_file, encoding_method
                )

                logger.start_timer()
                received_data = file_handler.read_file()
                decoded_metadata_str = channel.receive_and_decode(
                    received_data, encoding_method
                )
                decoded_metadata = caster.str_to_dict(decoded_metadata_str)
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
                    "mask_shape": caster.str_to_tuple(decoded_metadata["mask_shape"]),
                    "mask_non_zero_count": caster.str_to_int(
                        decoded_metadata["mask_non_zero_count"]
                    ),
                    "compressed_data": caster.str_to_ndarray(
                        decoded_metadata["compressed_data"]
                    ),
                }
                restored_image = inpainting.restore_image(decoded_metadata)
                logger.end_timer()

                output_image_file = os.path.join(
                    output_folder, f"{Path(image_file).stem}-{encoding_method}.png"
                )
                self.save_image(restored_image, output_image_file)

                iqa = IQA(np.array(image), restored_image)
                metrics = self.iqa_comparison(np.array(image), restored_image)

                print(f"IQA Results for {Path(image_file).stem}-{encoding_method}:")
                for metric, value in metrics.items():
                    print(f"{metric.upper()}: {value}")
                print(f"Processing time: {logger.processing_time:.2f} seconds\n")

    def process_images_with_different_quality_threshold(self) -> None:
        """
        Process images with different quality and threshold settings
        """
        image_files = self.get_image_files(self.input_folder)
        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        results = []

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            original_size = self.get_file_size(file_path)
            inpainting = Inpainting(image)

            for quality in range(0, 101, 10):
                for threshold in range(0, 256, 25):
                    for encoding_method in encoding_methods:
                        logger.start_timer()
                        metadata = inpainting.custom_compression(
                            quality=quality, threshold=threshold
                        )

                        metadata = caster.dict_to_str(
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
                            metadata, encoding_method
                        )

                        output_file = f"data/channel/{Path(image_file).stem}-{encoding_method}-quality{quality}-threshold{threshold}.txt"
                        file_handler = FileHandler(output_file)
                        file_handler.write_file(str(encoded_data))

                        encoded_size = self.get_file_size(output_file)
                        logger.end_timer()
                        processing_time = logger.processing_time

                        results.append(
                            {
                                "image_file": image_file,
                                "encoding_method": encoding_method,
                                "quality": quality,
                                "threshold": threshold,
                                "encoded_size": encoded_size,
                                "percentage_decrease": (
                                    (original_size - encoded_size) / original_size
                                )
                                * 100,
                                "processing_time": processing_time,
                            }
                        )

        self.display_results(results)

    def process_and_analyze_images_with_different_quality_threshold(self) -> None:
        """
        Analyze images processed with different quality and threshold settings
        """
        image_files = self.get_image_files(self.input_folder)
        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        results = []

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            original_size = self.get_file_size(file_path)
            inpainting = Inpainting(image)

            for quality in range(0, 101, 10):
                for threshold in range(0, 256, 25):
                    for encoding_method in encoding_methods:
                        logger.start_timer()
                        metadata = inpainting.custom_compression(
                            quality=quality, threshold=threshold
                        )

                        metadata = caster.dict_to_str(
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
                            metadata, encoding_method
                        )

                        output_file = f"data/channel/{Path(image_file).stem}-{encoding_method}-quality{quality}-threshold{threshold}.txt"
                        file_handler = FileHandler(output_file)
                        file_handler.write_file(str(encoded_data))

                        encoded_size = self.get_file_size(output_file)
                        logger.end_timer()
                        processing_time = logger.processing_time

                        results.append(
                            {
                                "image_file": image_file,
                                "encoding_method": encoding_method,
                                "quality": quality,
                                "threshold": threshold,
                                "encoded_size": encoded_size,
                                "percentage_decrease": (
                                    (original_size - encoded_size) / original_size
                                )
                                * 100,
                                "processing_time": processing_time,
                            }
                        )

        self.display_results(results)

    def process_and_restore_images_with_different_quality_threshold_by_iqa(
        self,
    ) -> None:
        """
        Process and restore images with different quality and threshold settings, and perform IQA
        """
        image_files = self.get_image_files(self.input_folder)
        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        output_folder = "data/restored"
        self.create_directory(output_folder)

        for image_file in tqdm(image_files, desc="Processing and restoring images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            original_size = self.get_file_size(file_path)
            inpainting = Inpainting(image)

            for quality in range(0, 101, 10):
                for threshold in range(0, 256, 25):
                    for encoding_method in encoding_methods:
                        logger.start_timer()
                        metadata = inpainting.custom_compression(
                            quality=quality, threshold=threshold
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
                        output_file = f"data/channel/{Path(image_file).stem}-{encoding_method}-quality{quality}-threshold{threshold}.txt"
                        file_handler = FileHandler(output_file)
                        file_handler.write_file(str(encoded_data))
                        encoded_size = self.get_file_size(output_file)
                        logger.end_timer()
                        self.log_process(
                            logger,
                            original_size,
                            encoded_size,
                            image_file,
                            encoding_method,
                        )

                        logger.start_timer()
                        received_data = file_handler.read_file()
                        decoded_metadata_str = channel.receive_and_decode(
                            received_data, encoding_method
                        )
                        decoded_metadata = caster.str_to_dict(decoded_metadata_str)
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
                        restored_image = inpainting.restore_image(decoded_metadata)
                        logger.end_timer()

                        output_image_file = os.path.join(
                            output_folder,
                            f"{Path(image_file).stem}-{encoding_method}-quality{quality}-threshold{threshold}.png",
                        )
                        self.save_image(restored_image, output_image_file)

                        iqa = IQA(np.array(image), restored_image)
                        metrics = self.iqa_comparison(np.array(image), restored_image)

                        print(
                            f"IQA Results for {Path(image_file).stem}-{encoding_method}-quality{quality}-threshold{threshold}:"
                        )
                        for metric, value in metrics.items():
                            print(f"{metric.upper()}: {value}")
                        print(
                            f"Processing time: {logger.processing_time:.2f} seconds\n"
                        )

    def process_and_restored_analyze_images_with_different_quality_threshold(
        self,
    ) -> None:
        """
        Analyze processed and restored images with different quality and threshold settings
        """
        image_files = self.get_image_files(self.input_folder)
        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        output_folder = "data/restored"
        self.create_directory(output_folder)

        results = []

        for image_file in tqdm(image_files, desc="Processing and restoring images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            original_size = self.get_file_size(file_path)
            inpainting = Inpainting(image)

            for quality in range(0, 101, 10):
                for threshold in range(0, 256, 25):
                    for encoding_method in encoding_methods:
                        logger.start_timer()
                        metadata = inpainting.custom_compression(
                            quality=quality, threshold=threshold
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
                        output_file = f"data/channel/{Path(image_file).stem}-{encoding_method}-quality{quality}-threshold{threshold}.txt"
                        file_handler = FileHandler(output_file)
                        file_handler.write_file(str(encoded_data))
                        encoded_size = self.get_file_size(output_file)
                        logger.end_timer()
                        processing_time = logger.processing_time
                        percentage_decrease = (
                            (original_size - encoded_size) / original_size
                        ) * 100

                        logger.start_timer()
                        received_data = file_handler.read_file()
                        decoded_metadata_str = channel.receive_and_decode(
                            received_data, encoding_method
                        )
                        decoded_metadata = caster.str_to_dict(decoded_metadata_str)
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
                        restored_image = inpainting.restore_image(decoded_metadata)
                        logger.end_timer()

                        output_image_file = os.path.join(
                            output_folder,
                            f"{Path(image_file).stem}-{encoding_method}-quality{quality}-threshold{threshold}.png",
                        )
                        self.save_image(restored_image, output_image_file)

                        iqa = IQA(np.array(image), restored_image)
                        metrics = self.iqa_comparison(np.array(image), restored_image)

                        results.append(
                            {
                                "image_file": image_file,
                                "encoding_method": encoding_method,
                                "quality": quality,
                                "threshold": threshold,
                                "percentage_decrease": percentage_decrease,
                                "processing_time": processing_time,
                                "mse": metrics["mse"],
                                "psnr": metrics["psnr"],
                                "ssim": metrics["ssim"],
                                "rmse": metrics["rmse"],
                                "mae": metrics["mae"],
                                "entropy": metrics["entropy"],
                                "compression_ratio": metrics["compression_ratio"],
                                "bitrate": metrics["bitrate"],
                                "fsim": metrics["fsim"],
                            }
                        )

        self.display_results(results)

    def process_images_with_different_quality_threshold_edge_methods(self) -> None:
        """
        Process images with different edge detection methods, quality, and threshold settings
        """
        image_files = self.get_image_files(self.input_folder)
        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
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
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        results = []

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            original_size = self.get_file_size(file_path)
            inpainting = Inpainting(image)

            for quality in range(0, 101, 10):
                for threshold in range(0, 256, 25):
                    for edge_method in edge_methods:
                        for encoding_method in encoding_methods:
                            logger.start_timer()
                            metadata = inpainting.custom_compression(
                                quality=quality,
                                threshold=threshold,
                                method="high_intensity_and_edge",
                            )

                            metadata = caster.dict_to_str(
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
                                metadata, encoding_method
                            )

                            output_file = f"data/channel/{Path(image_file).stem}-{encoding_method}-edge_{edge_method}-quality{quality}-threshold{threshold}.txt"

                            file_handler = FileHandler(output_file)
                            file_handler.write_file(str(encoded_data))

                            encoded_size = self.get_file_size(output_file)
                            logger.end_timer()
                            processing_time = logger.processing_time

                            results.append(
                                {
                                    "image_file": image_file,
                                    "encoding_method": encoding_method,
                                    "edge_method": edge_method,
                                    "quality": quality,
                                    "threshold": threshold,
                                    "encoded_size": encoded_size,
                                    "percentage_decrease": (
                                        (original_size - encoded_size) / original_size
                                    )
                                    * 100,
                                    "processing_time": processing_time,
                                }
                            )

        self.display_results(results)

    def process_and_restore_images_with_different_quality_threshold_edge_methods(
        self,
    ) -> None:
        """
        Restore processed images with different edge detection methods, quality, and threshold settings
        """
        encoded_files = self.get_encoded_files(
            "data/channel", "*-edge_*-quality*-threshold*.txt"
        )
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        output_folder = "data/restored"
        self.create_directory(output_folder)

        results = []

        for encoded_file in tqdm(encoded_files, desc="Processing files"):
            encoding_method = Path(encoded_file).stem.split("-")[1]
            edge_method = Path(encoded_file).stem.split("-")[2].split("_")[1]
            quality = int(Path(encoded_file).stem.split("-")[3].split("quality")[1])
            threshold = int(Path(encoded_file).stem.split("-")[4].split("threshold")[1])

            file_handler = FileHandler(encoded_file)
            encoded_data = file_handler.read_file()

            logger.start_timer()
            metadata = channel.receive_and_decode(encoded_data, encoding_method)
            decoded_metadata = caster.str_to_dict(metadata)
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
                "mask_shape": caster.str_to_tuple(decoded_metadata["mask_shape"]),
                "mask_non_zero_count": caster.str_to_int(
                    decoded_metadata["mask_non_zero_count"]
                ),
                "compressed_data": caster.str_to_ndarray(
                    decoded_metadata["compressed_data"]
                ),
            }

            inpainting = Inpainting()
            restored_image = inpainting.restore_image(decoded_metadata)
            logger.end_timer()

            output_image_file = f"{Path(encoded_file).stem}.png"
            self.save_image(
                restored_image, os.path.join(output_folder, output_image_file)
            )

            original_image_path = os.path.join(
                self.input_folder, f"{Path(encoded_file).stem.split('-')[0]}.png"
            )
            original_image = ImageProcessing().open(original_image_path)
            iqa = IQA(np.array(original_image), restored_image)
            metrics = self.iqa_comparison(np.array(original_image), restored_image)

            results.append(
                {
                    "image_file": original_image_path,
                    "encoding_method": encoding_method,
                    "edge_method": edge_method,
                    "quality": quality,
                    "threshold": threshold,
                    "percentage_decrease": (
                        (
                            self.get_file_size(original_image_path)
                            - self.get_file_size(output_image_file)
                        )
                        / self.get_file_size(original_image_path)
                    )
                    * 100,
                    "processing_time": logger.processing_time,
                    "mse": metrics["mse"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "entropy": metrics["entropy"],
                    "compression_ratio": metrics["compression_ratio"],
                    "bitrate": metrics["bitrate"],
                    "fsim": metrics["fsim"],
                }
            )

        self.display_results(results)

    def process_mask_images_with_different_parameters(self) -> None:
        """
        Process mask images using various edge detection methods and thresholds
        """
        output_folder = "data/mask"
        self.create_directory(output_folder)
        image_files = self.get_image_files(self.input_folder)

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

        thresholds = [50, 100, 150, 200]

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            img = ImageProcessing()
            original_image = img.open(file_path)

            inpainting_processor = Inpainting(original_image)

            for edge_method in edge_methods:
                for threshold in thresholds:
                    mask_high_intensity_and_edge = inpainting_processor.select_removable_area_by_high_intensity_and_edge(
                        threshold=threshold, edge_method=edge_method
                    )
                    mask_path = os.path.join(
                        output_folder,
                        f"{self.get_base_name(image_file)}-high_intensity_edge_{edge_method}_threshold_{threshold}.png",
                    )
                    self.save_image(mask_high_intensity_and_edge, mask_path)

    def process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path(
        self,
    ) -> None:
        """
        Analyze images with different quality, threshold, edge detection methods, and inpainting parameters
        """
        image_files = self.get_image_files(self.input_folder)
        encoding_methods = ["rle", "hamming", "huffman", "base64", "ascii"]
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
        inpaint_methods = ["telea", "ns"]
        inpaint_radii = range(3, 23, 2)
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        results = []

        for image_file in tqdm(image_files, desc="Processing images"):
            file_path = os.path.join(self.input_folder, image_file)
            image = ImageProcessing().open(file_path)
            original_size = self.get_file_size(file_path)
            inpainting = Inpainting(image)

            for quality in range(0, 101, 10):
                for threshold in range(0, 256, 25):
                    for edge_method in edge_methods:
                        for encoding_method in encoding_methods:
                            for inpaint_method in inpaint_methods:
                                for inpaint_radius in inpaint_radii:
                                    logger.start_timer()
                                    metadata = inpainting.custom_compression(
                                        quality=quality,
                                        threshold=threshold,
                                        method="high_intensity_and_edge",
                                        inpaint_method=inpaint_method,
                                        inpaint_radius=inpaint_radius,
                                    )

                                    metadata = caster.dict_to_str(
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
                                        metadata, encoding_method
                                    )

                                    output_file = f"data/channel/{Path(image_file).stem}-{encoding_method}-edge_{edge_method}-quality{quality}-threshold{threshold}-inpaint_{inpaint_method}-radius_{inpaint_radius}.txt"

                                    file_handler = FileHandler(output_file)
                                    file_handler.write_file(str(encoded_data))

                                    encoded_size = self.get_file_size(output_file)
                                    logger.end_timer()
                                    processing_time = logger.processing_time

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
                                        }
                                    )

        self.display_results(results)

    def process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path(
        self,
    ) -> None:
        """
        Restore images with different quality, threshold, edge detection methods, and inpainting parameters
        """
        encoded_files = self.get_encoded_files(
            "data/channel", "*-edge_*-quality*-threshold*.txt"
        )
        channel = Channel()
        caster = TypeCaster()
        logger = ProcessLogger()

        output_folder = "data/restored"
        self.create_directory(output_folder)

        results = []

        for encoded_file in tqdm(encoded_files, desc="Processing files"):
            encoding_method = Path(encoded_file).stem.split("-")[1]
            edge_method = Path(encoded_file).stem.split("-")[2].split("_")[1]
            quality = int(Path(encoded_file).stem.split("-")[3].split("quality")[1])
            threshold = int(Path(encoded_file).stem.split("-")[4].split("threshold")[1])
            inpaint_method = Path(encoded_file).stem.split("-")[5].split("inpaint_")[1]
            inpaint_radius = int(
                Path(encoded_file).stem.split("-")[6].split("radius_")[1]
            )

            file_handler = FileHandler(encoded_file)
            encoded_data = file_handler.read_file()

            logger.start_timer()
            metadata = channel.receive_and_decode(encoded_data, encoding_method)
            decoded_metadata = caster.str_to_dict(metadata)
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
                "mask_shape": caster.str_to_tuple(decoded_metadata["mask_shape"]),
                "mask_non_zero_count": caster.str_to_int(
                    decoded_metadata["mask_non_zero_count"]
                ),
                "compressed_data": caster.str_to_ndarray(
                    decoded_metadata["compressed_data"]
                ),
            }

            inpainting = Inpainting()
            restored_image = inpainting.restore_image(
                decoded_metadata,
                inpaint_method=inpaint_method,
                inpaint_radius=inpaint_radius,
            )
            logger.end_timer()

            output_image_file = f"{Path(encoded_file).stem}.png"
            self.save_image(
                restored_image, os.path.join(output_folder, output_image_file)
            )

            original_image_path = os.path.join(
                self.input_folder, f"{Path(encoded_file).stem.split('-')[0]}.png"
            )
            original_image = ImageProcessing().open(original_image_path)
            iqa = IQA(np.array(original_image), restored_image)
            metrics = self.iqa_comparison(np.array(original_image), restored_image)

            results.append(
                {
                    "image_file": original_image_path,
                    "encoding_method": encoding_method,
                    "edge_method": edge_method,
                    "quality": quality,
                    "threshold": threshold,
                    "inpaint_method": inpaint_method,
                    "inpaint_radius": inpaint_radius,
                    "percentage_decrease": (
                        (
                            self.get_file_size(original_image_path)
                            - self.get_file_size(output_image_file)
                        )
                        / self.get_file_size(original_image_path)
                    )
                    * 100,
                    "processing_time": logger.processing_time,
                    "mse": metrics["mse"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "entropy": metrics["entropy"],
                    "compression_ratio": metrics["compression_ratio"],
                    "bitrate": metrics["bitrate"],
                    "fsim": metrics["fsim"],
                }
            )

        self.display_results(results)

    def highlight_image_by_mask(self) -> None:
        mask_output_folder = "data/mask"
        highlight_output_folder = "data/highlight"
        self.create_directory(mask_output_folder)
        self.create_directory(highlight_output_folder)
        image_files = self.get_image_files(self.input_folder)
        for image_file in tqdm(image_files, desc="Processing images"):
            img = ImageProcessing()

            file_path = os.path.join(self.input_folder, image_file)
            original_image = img.open(file_path)

            inpainting_processor = Inpainting(original_image)

            # Process different masks and highlight images accordingly
            mask_types = [
                (
                    "high_intensity",
                    inpainting_processor.select_removable_area_by_high_intensity,
                ),
                (
                    "high_intensity_edge_canny",
                    lambda: inpainting_processor.select_removable_area_by_high_intensity_and_edge(
                        edge_method="canny"
                    ),
                ),
                (
                    "high_intensity_edge_optimal_canny",
                    lambda: inpainting_processor.select_removable_area_by_high_intensity_and_edge(
                        edge_method="optimal_canny"
                    ),
                ),
            ]

            for mask_name, mask_function in mask_types:
                mask = mask_function()
                mask_path = os.path.join(
                    mask_output_folder,
                    f"{self.get_base_name(image_file)}-{mask_name}.png",
                )
                self.save_image(mask, mask_path)

                # Highlight the image using the generated mask
                highlighted_image = img.highlight_image(original_image, mask)
                highlighted_image_path = os.path.join(
                    highlight_output_folder,
                    f"{self.get_base_name(image_file)}-{mask_name}-highlighted.png",
                )

                from PIL import Image

                highlighted_image_pil = Image.fromarray(highlighted_image)
                highlighted_image_pil.save(highlighted_image_path)


if __name__ == "__main__":
    main = Main()

    # 1. Clear the project workspace
    main.clear()

    # # 2. Perform edge detection on benchmark images
    # main.edge_detection()

    # # 3. Apply JPEG compression to images
    # main.jpeg_compression()

    # # 4. Compress images using various compression methods and store the result
    # images = main.image_compression()

    # # 5. Conduct image quality assessment (IQA) by comparing original and compressed images
    # main.iqa_original_by_compressed()

    # # 6. Process images to create masks using different parameters
    # main.process_mask_images()

    # # 7. Perform inpainting on images and inpainting
    # main.inpainting()

    # # 8. Perform inpainting on images and inpainting using edge detection
    # main.inpainting_by_edge()

    # # 9. Apply various encoding algorithms to a text string and verify correctness
    # main.encoding_algorithms("I like Python")
    # main.encoding_algorithms_by_channel("I like Python")

    # # 10. Verify image conversion by converting to string and back to ndarray
    # main.verify_image_conversion()

    # # 11. Process images without inpainting
    # main.process_images_without_inpainting()

    # # 12. Restore images processed without inpainting
    # main.restore_process_images_without_inpainting()

    # # 13. Process images with inpainting and transmit through channel encoding
    # main.process_images()

    # # 14. Restore images processed with inpainting and transmitted through channel encoding
    # main.restore_process_images()

    # # 15. Process and restore inpainting images with image quality assessment (IQA)
    # main.process_and_restore_inpainting_by_iqa()

    # # 16. Process images with different quality and threshold settings
    # main.process_images_with_different_quality_threshold()

    # # 17. Analyze images processed with different quality and threshold settings
    # main.process_and_analyze_images_with_different_quality_threshold()

    # # 18. Process and restore images with different quality and threshold settings, and perform IQA
    # main.process_and_restore_images_with_different_quality_threshold_by_iqa()

    # # 19. Analyze processed and restored images with different quality and threshold settings
    # main.process_and_restored_analyze_images_with_different_quality_threshold()

    # # 20. Process images with different edge detection methods, quality, and threshold settings
    # main.process_images_with_different_quality_threshold_edge_methods()

    # # 21. Analyze processed images with different edge detection methods, quality, and threshold settings
    # main.process_and_analyze_images_with_different_quality_threshold_edge_methods()

    # # 22. Restore processed images with different edge detection methods, quality, and threshold settings
    # main.process_and_restore_images_with_different_quality_threshold_edge_methods()

    # # 23. Process mask images using various edge detection methods and thresholds
    # main.process_mask_images_with_different_parameters()

    # # 24. Analyze images with different quality, threshold, edge detection methods, and inpainting parameters
    # main.process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path()

    # # 25. Restore images with different quality, threshold, edge detection methods, and inpainting parameters
    # main.process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path()

    # # 26. Highlights an image based on the mask values.
    # main.highlight_image_by_mask()
