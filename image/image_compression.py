import pillow_avif
import numpy as np
from PIL import Image
from io import BytesIO


class ImageCompression:
    def __init__(self, image: np.ndarray | None = None) -> None:
        """
        Initializes the class instance.
        """
        self.image = image

    # 1. JPEG Compression
    # Reference: Wallace, G. K. (1991). The JPEG still picture compression standard. Communications of the ACM.
    def compress_jpeg(self, quality: int = 90) -> np.ndarray:
        """
        Compresses an image using JPEG algorithm with a specified quality level.
        """
        import cv2

        if self.image is None:
            raise ValueError("Image is not available for compression.")

        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode(".jpg", self.image, encode_param)
            if result:
                return encimg
            else:
                raise ValueError("JPEG compression failed")
        except Exception as e:
            raise Exception(f"Failed to compress JPEG: {e}")

    def decompress_jpeg(self, data: np.ndarray) -> np.ndarray:
        """
        Decompresses JPEG-compressed image data into its original form.
        """
        import cv2

        try:
            np_data = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if image is not None:
                return image
            else:
                raise ValueError("JPEG decompression failed")
        except Exception as e:
            raise Exception(f"Failed to decompress JPEG: {e}")

    # 2. PNG Compression
    # Reference: Boutell, T. (1997). PNG (Portable Network Graphics) specification. Version 1.0.
    def compress_png(self, compression_level: int = 3) -> np.ndarray:
        """
        Compresses an image using PNG algorithm with a specified compression level.
        """
        import cv2

        if self.image is None:
            raise ValueError("Image is not available for compression.")

        try:
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level]
            result, encimg = cv2.imencode(".png", self.image, encode_param)
            if result:
                return encimg
            else:
                raise ValueError("PNG compression failed")
        except Exception as e:
            raise Exception(f"Failed to compress PNG: {e}")

    # 3. WebP Compression
    # Reference: Google. (2010). WebP: A new image format for the web.
    def compress_webp(self, quality: int = 90) -> np.ndarray:
        """
        Compresses an image using WebP algorithm with a specified quality level.
        """
        import cv2

        if self.image is None:
            raise ValueError("Image is not available for compression.")

        try:
            encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
            result, encimg = cv2.imencode(".webp", self.image, encode_param)
            if result:
                return encimg
            else:
                raise ValueError("WebP compression failed")
        except Exception as e:
            raise Exception(f"Failed to compress WebP: {e}")

    # 4. TIFF Compression
    # Reference: Adobe Developers Association. (1992). TIFF Revision 6.0 Final.
    def compress_tiff(self, compression_type: int = 1) -> np.ndarray:
        """
        Compresses an image using TIFF algorithm with a specified compression type.
        """
        import cv2

        if self.image is None:
            raise ValueError("Image is not available for compression.")

        try:
            encode_param = [int(cv2.IMWRITE_TIFF_COMPRESSION), compression_type]
            result, encimg = cv2.imencode(".tiff", self.image, encode_param)
            if result:
                return encimg
            else:
                raise ValueError("TIFF compression failed")
        except Exception as e:
            raise Exception(f"Failed to compress TIFF: {e}")

    # 5. JPEG2000 Compression
    # Reference: Taubman, D. S., & Marcellin, M. W. (2001). JPEG2000: Image compression fundamentals, standards and practice.
    def compress_jpeg2000(self, quality: int = 90) -> np.ndarray:
        """
        Compresses an image using JPEG2000 algorithm with a specified quality level.
        """
        import cv2

        if self.image is None:
            raise ValueError("Image is not available for compression.")

        try:
            encode_param = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), quality * 100]
            result, encimg = cv2.imencode(".jp2", self.image, encode_param)
            if result:
                return encimg
            else:
                raise ValueError("JPEG2000 compression failed")
        except Exception as e:
            raise Exception(f"Failed to compress JPEG2000: {e}")

    # 6. AVIF Compression
    # Reference: "AV1 Image Compression" by the Alliance for Open Media (AOMedia) in 2018.
    def compress_avif(self, quality: int = 90) -> bytes:
        """
        Compresses an image using AVIF algorithm with a specified quality level.
        """
        import cv2

        if self.image is None:
            raise ValueError("Image is not available for compression.")

        try:
            # Convert BGR to RGB if necessary
            if self.image.shape[2] == 3:  # Ensure the image has three channels
                self.image = self.image[..., ::-1]  # Reverse the channels

            img = Image.fromarray(self.image)
            if img.mode != "RGB":
                img = img.convert("RGB")
            output = BytesIO()
            img.save(output, format="AVIF", quality=quality)
            return output.getvalue()
        except Exception as e:
            raise Exception(f"Failed to compress AVIF: {e}")

    def decompress_and_decode(self, data: np.ndarray) -> np.ndarray:
        """
        Decompress and decode an image.
        """
        import cv2
        import numpy as np

        data = data.tobytes()
        data = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
