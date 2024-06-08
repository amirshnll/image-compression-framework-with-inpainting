# Advanced Image Compression Framework with Integrated Inpainting and Multi-Algorithm Encoding Techniques

A powerful and sophisticated image compression framework that combines advanced data encoding techniques such as RLE, Hamming, Huffman, Base64, and ASCII with edge detection and inpainting methods like Sobel, Prewitt, Roberts, LoG, Canny, Scharr, Zero Crossing, and Optimal Canny. Utilizing multi-radius, multi-threshold, and multi-quality settings, this comprehensive suite supports selecting high-intensity regions, applying Telea and NS inpainting methods, and includes tools for edge-aware inpainting, dynamic compression adjustments, custom compression algorithms, and robust image quality assessment, ensuring efficient and high-performance compression across multiple image formats.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run the code in this repository, you need to have Python installed. Additionally, install the required packages using the following command:

```sh
pip install -r requirements.txt
```

## Usage
To use this framework, you can execute the main script:

```sh
python main.py
```
This will run all the defined processes sequentially. You can modify the main.py script to run specific functions as needed.

## Features
- Edge Detection: Applies various edge detection methods to images.
- JPEG Compression: Compresses images using JPEG compression and evaluates the results.
- Image Compression: Compresses images using multiple compression methods (JPEG, PNG, WebP, TIFF, JPEG2000, AVIF).
- Image Quality Assessment (IQA): Compares original and compressed images using metrics like MSE, PSNR, SSIM, etc.
- Inpainting: Fills in missing parts of images using different inpainting techniques.
- Encoding Algorithms: Applies various encoding algorithms to text strings and verifies correctness.
- Image Conversion: Converts images to strings and back to ndarray format.
- Processing with Different Parameters: Processes images with different quality and threshold settings, including edge detection methods.

## Contributing
Contributions are welcome! Please fork this repository and submit pull requests for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.

## Note
Make sure to adjust the `requirements.txt` installation command and any other specific instructions according to your project's actual setup. 