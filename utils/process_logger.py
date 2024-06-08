import os
import time


class ProcessLogger:
    """
    The ProcessLogger class is designed to facilitate the logging of processing times
    and efficiency metrics for various encoding methods applied to image files.
    This class provides methods to start and stop a timer, calculate the processing time,
    and log relevant details of the process.
    """

    def __init__(self) -> None:
        self.start_time: None
        self.end_time: None

    def start_timer(self) -> None:
        self.start_time = time.time()

    def end_timer(self) -> None:
        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time

    def log_process(
        self,
        original_size: int,
        encoded_size: int,
        image_file: str,
        encoding_method: str,
    ) -> None:
        if self.start_time is None:
            self.start_timer()
        if self.end_time is None:
            self.end_timer()

        percentage_decrease = ((original_size - encoded_size) / original_size) * 100
        print(f"Processed {image_file} with {encoding_method} encoding")
        print(
            f"Original size: {original_size} bytes, Encoded size: {encoded_size} bytes, "
            f"Percentage decrease: {percentage_decrease:.2f}%, "
            f"Processing time: {self.processing_time:.2f} seconds"
        )
