import os
from utils import Flowchart, ClearProject
from typing import Dict, List, Tuple

# Clear flowchart directory
ClearProject().clear_data_directory("data/flowchart")

# Ensure the output directory exists
os.makedirs("data/flowchart", exist_ok=True)

# Draw flowcharts


def draw_flowchart(
    title: str,
    nodes: Dict[str, str],
    edges: List[Tuple[str, str]],
    subgraphs: Dict[str, Dict[str, str]],
    filename: str,
) -> None:
    drawer = Flowchart(title)
    drawer.add_nodes(nodes)
    drawer.add_edges(edges)
    drawer.add_subgraphs(subgraphs)
    drawer.render(filename)


# main.edge_detection()
nodes_edge_detection = {
    "LoadImage": "Load image",
    "ConvertGray": "Convert to grayscale",
    "ApplySobel": "Apply Sobel filter",
    "ApplyCanny": "Apply Canny edge detection",
    "SaveResult": "Save result",
}
edges_edge_detection = [
    ("LoadImage", "ConvertGray"),
    ("ConvertGray", "ApplySobel"),
    ("ApplySobel", "ApplyCanny"),
    ("ApplyCanny", "SaveResult"),
]
subgraphs_edge_detection = {
    "cluster_edge_detection": {
        "label": "Edge Detection Process",
        "nodes": nodes_edge_detection,
    }
}
draw_flowchart(
    "Edge Detection",
    nodes_edge_detection,
    edges_edge_detection,
    subgraphs_edge_detection,
    "edge_detection",
)

# main.jpeg_compression()
nodes_jpeg_compression = {
    "LoadImage": "Load image",
    "CompressJPEG": "Compress using JPEG",
    "SaveCompressed": "Save compressed image",
}
edges_jpeg_compression = [
    ("LoadImage", "CompressJPEG"),
    ("CompressJPEG", "SaveCompressed"),
]
subgraphs_jpeg_compression = {
    "cluster_jpeg_compression": {
        "label": "JPEG Compression Process",
        "nodes": nodes_jpeg_compression,
    }
}
draw_flowchart(
    "JPEG Compression",
    nodes_jpeg_compression,
    edges_jpeg_compression,
    subgraphs_jpeg_compression,
    "jpeg_compression",
)

# main.image_compression()
nodes_image_compression = {
    "LoadImage": "Load image",
    "CompressPNG": "Compress using PNG",
    "CompressWebP": "Compress using WebP",
    "SaveCompressed": "Save compressed images",
}
edges_image_compression = [
    ("LoadImage", "CompressPNG"),
    ("CompressPNG", "CompressWebP"),
    ("CompressWebP", "SaveCompressed"),
]
subgraphs_image_compression = {
    "cluster_image_compression": {
        "label": "Image Compression Process",
        "nodes": nodes_image_compression,
    }
}
draw_flowchart(
    "Image Compression",
    nodes_image_compression,
    edges_image_compression,
    subgraphs_image_compression,
    "image_compression",
)

# main.iqa_original_by_compressed()
nodes_iqa = {
    "LoadOriginal": "Load original image",
    "LoadCompressed": "Load compressed image",
    "CalculateMSE": "Calculate MSE",
    "CalculatePSNR": "Calculate PSNR",
    "CalculateSSIM": "Calculate SSIM",
    "SaveResults": "Save results",
}
edges_iqa = [
    ("LoadOriginal", "LoadCompressed"),
    ("LoadCompressed", "CalculateMSE"),
    ("CalculateMSE", "CalculatePSNR"),
    ("CalculatePSNR", "CalculateSSIM"),
    ("CalculateSSIM", "SaveResults"),
]
subgraphs_iqa = {"cluster_iqa": {"label": "IQA Process", "nodes": nodes_iqa}}
draw_flowchart(
    "IQA Original by Compressed",
    nodes_iqa,
    edges_iqa,
    subgraphs_iqa,
    "iqa_original_by_compressed",
)

# main.process_mask_images()
nodes_process_mask_images = {
    "LoadImage": "Load image",
    "ApplyThreshold": "Apply threshold",
    "CreateMask": "Create mask",
    "SaveMask": "Save mask",
}
edges_process_mask_images = [
    ("LoadImage", "ApplyThreshold"),
    ("ApplyThreshold", "CreateMask"),
    ("CreateMask", "SaveMask"),
]
subgraphs_process_mask_images = {
    "cluster_process_mask_images": {
        "label": "Process Mask Images",
        "nodes": nodes_process_mask_images,
    }
}
draw_flowchart(
    "Process Mask Images",
    nodes_process_mask_images,
    edges_process_mask_images,
    subgraphs_process_mask_images,
    "process_mask_images",
)

# main.inpainting()
nodes_inpainting = {
    "LoadImage": "Load image",
    "LoadMask": "Load mask",
    "ApplyInpainting": "Apply inpainting",
    "SaveInpaintedImage": "Save inpainted image",
}
edges_inpainting = [
    ("LoadImage", "LoadMask"),
    ("LoadMask", "ApplyInpainting"),
    ("ApplyInpainting", "SaveInpaintedImage"),
]
subgraphs_inpainting = {
    "cluster_inpainting": {"label": "Inpainting Process", "nodes": nodes_inpainting}
}
draw_flowchart(
    "Inpainting", nodes_inpainting, edges_inpainting, subgraphs_inpainting, "inpainting"
)

# main.inpainting_by_edge()
nodes_inpainting_by_edge = {
    "LoadImage": "Load image",
    "DetectEdges": "Detect edges",
    "CreateMaskFromEdges": "Create mask from edges",
    "ApplyInpainting": "Apply inpainting",
    "SaveInpaintedImage": "Save inpainted image",
}
edges_inpainting_by_edge = [
    ("LoadImage", "DetectEdges"),
    ("DetectEdges", "CreateMaskFromEdges"),
    ("CreateMaskFromEdges", "ApplyInpainting"),
    ("ApplyInpainting", "SaveInpaintedImage"),
]
subgraphs_inpainting_by_edge = {
    "cluster_inpainting_by_edge": {
        "label": "Inpainting by Edge Detection",
        "nodes": nodes_inpainting_by_edge,
    }
}
draw_flowchart(
    "Inpainting by Edge Detection",
    nodes_inpainting_by_edge,
    edges_inpainting_by_edge,
    subgraphs_inpainting_by_edge,
    "inpainting_by_edge",
)

# main.encoding_algorithms()
nodes_encoding_algorithms = {
    "InputString": "Input string",
    "ApplyRLE": "Apply RLE",
    "ApplyHamming": "Apply Hamming",
    "ApplyHuffman": "Apply Huffman",
    "ApplyBase64": "Apply Base64",
    "ApplyASCII": "Apply ASCII",
    "VerifyEncoding": "Verify encoding",
}
edges_encoding_algorithms = [
    ("InputString", "ApplyRLE"),
    ("ApplyRLE", "ApplyHamming"),
    ("ApplyHamming", "ApplyHuffman"),
    ("ApplyHuffman", "ApplyBase64"),
    ("ApplyBase64", "ApplyASCII"),
    ("ApplyASCII", "VerifyEncoding"),
]
subgraphs_encoding_algorithms = {
    "cluster_encoding_algorithms": {
        "label": "Encoding Algorithms",
        "nodes": nodes_encoding_algorithms,
    }
}
draw_flowchart(
    "Encoding Algorithms",
    nodes_encoding_algorithms,
    edges_encoding_algorithms,
    subgraphs_encoding_algorithms,
    "encoding_algorithms",
)

# main.encoding_algorithms_by_channel()
nodes_encoding_algorithms_by_channel = {
    "InputString": "Input string",
    "EncodeAndTransmit": "Encode and transmit",
    "ReceiveAndDecode": "Receive and decode",
    "VerifyEncoding": "Verify encoding",
}
edges_encoding_algorithms_by_channel = [
    ("InputString", "EncodeAndTransmit"),
    ("EncodeAndTransmit", "ReceiveAndDecode"),
    ("ReceiveAndDecode", "VerifyEncoding"),
]
subgraphs_encoding_algorithms_by_channel = {
    "cluster_encoding_algorithms_by_channel": {
        "label": "Encoding Algorithms by Channel",
        "nodes": nodes_encoding_algorithms_by_channel,
    }
}
draw_flowchart(
    "Encoding Algorithms by Channel",
    nodes_encoding_algorithms_by_channel,
    edges_encoding_algorithms_by_channel,
    subgraphs_encoding_algorithms_by_channel,
    "encoding_algorithms_by_channel",
)

# main.verify_image_conversion()
nodes_verify_image_conversion = {
    "LoadImage": "Load image",
    "ConvertToString": "Convert to string",
    "ConvertToNDArray": "Convert to ndarray",
    "VerifyConversion": "Verify conversion",
}
edges_verify_image_conversion = [
    ("LoadImage", "ConvertToString"),
    ("ConvertToString", "ConvertToNDArray"),
    ("ConvertToNDArray", "VerifyConversion"),
]
subgraphs_verify_image_conversion = {
    "cluster_verify_image_conversion": {
        "label": "Verify Image Conversion",
        "nodes": nodes_verify_image_conversion,
    }
}
draw_flowchart(
    "Verify Image Conversion",
    nodes_verify_image_conversion,
    edges_verify_image_conversion,
    subgraphs_verify_image_conversion,
    "verify_image_conversion",
)

# main.process_images_without_inpainting()
nodes_process_images_without_inpainting = {
    "LoadImage": "Load image",
    "CompressImage": "Compress image",
    "SaveCompressedImage": "Save compressed image",
}
edges_process_images_without_inpainting = [
    ("LoadImage", "CompressImage"),
    ("CompressImage", "SaveCompressedImage"),
]
subgraphs_process_images_without_inpainting = {
    "cluster_process_images_without_inpainting": {
        "label": "Process Images Without Inpainting",
        "nodes": nodes_process_images_without_inpainting,
    }
}
draw_flowchart(
    "Process Images Without Inpainting",
    nodes_process_images_without_inpainting,
    edges_process_images_without_inpainting,
    subgraphs_process_images_without_inpainting,
    "process_images_without_inpainting",
)

# main.restore_process_images_without_inpainting()
nodes_restore_process_images_without_inpainting = {
    "LoadCompressedImage": "Load compressed image",
    "DecompressImage": "Decompress image",
    "SaveRestoredImage": "Save restored image",
}
edges_restore_process_images_without_inpainting = [
    ("LoadCompressedImage", "DecompressImage"),
    ("DecompressImage", "SaveRestoredImage"),
]
subgraphs_restore_process_images_without_inpainting = {
    "cluster_restore_process_images_without_inpainting": {
        "label": "Restore Process Images Without Inpainting",
        "nodes": nodes_restore_process_images_without_inpainting,
    }
}
draw_flowchart(
    "Restore Process Images Without Inpainting",
    nodes_restore_process_images_without_inpainting,
    edges_restore_process_images_without_inpainting,
    subgraphs_restore_process_images_without_inpainting,
    "restore_process_images_without_inpainting",
)

# main.process_images()
nodes_process_images = {
    "LoadImage": "Load image",
    "CreateMask": "Create mask",
    "ApplyInpainting": "Apply inpainting",
    "CompressImage": "Compress image",
    "SaveCompressedImage": "Save compressed image",
}
edges_process_images = [
    ("LoadImage", "CreateMask"),
    ("CreateMask", "ApplyInpainting"),
    ("ApplyInpainting", "CompressImage"),
    ("CompressImage", "SaveCompressedImage"),
]
subgraphs_process_images = {
    "cluster_process_images": {"label": "Process Images", "nodes": nodes_process_images}
}
draw_flowchart(
    "Process Images",
    nodes_process_images,
    edges_process_images,
    subgraphs_process_images,
    "process_images",
)

# main.restore_process_images()
nodes_restore_process_images = {
    "LoadCompressedImage": "Load compressed image",
    "DecompressImage": "Decompress image",
    "RestoreImage": "Restore image",
    "SaveRestoredImage": "Save restored image",
}
edges_restore_process_images = [
    ("LoadCompressedImage", "DecompressImage"),
    ("DecompressImage", "RestoreImage"),
    ("RestoreImage", "SaveRestoredImage"),
]
subgraphs_restore_process_images = {
    "cluster_restore_process_images": {
        "label": "Restore Process Images",
        "nodes": nodes_restore_process_images,
    }
}
draw_flowchart(
    "Restore Process Images",
    nodes_restore_process_images,
    edges_restore_process_images,
    subgraphs_restore_process_images,
    "restore_process_images",
)

# main.process_and_restore_inpainting_by_iqa()
nodes_process_and_restore_inpainting_by_iqa = {
    "LoadImage": "Load image",
    "CreateMask": "Create mask",
    "ApplyInpainting": "Apply inpainting",
    "CompressImage": "Compress image",
    "DecompressImage": "Decompress image",
    "RestoreImage": "Restore image",
    "CalculateIQA": "Calculate IQA",
    "SaveResults": "Save results",
}
edges_process_and_restore_inpainting_by_iqa = [
    ("LoadImage", "CreateMask"),
    ("CreateMask", "ApplyInpainting"),
    ("ApplyInpainting", "CompressImage"),
    ("CompressImage", "DecompressImage"),
    ("DecompressImage", "RestoreImage"),
    ("RestoreImage", "CalculateIQA"),
    ("CalculateIQA", "SaveResults"),
]
subgraphs_process_and_restore_inpainting_by_iqa = {
    "cluster_process_and_restore_inpainting_by_iqa": {
        "label": "Process and Restore Inpainting by IQA",
        "nodes": nodes_process_and_restore_inpainting_by_iqa,
    }
}
draw_flowchart(
    "Process and Restore Inpainting by IQA",
    nodes_process_and_restore_inpainting_by_iqa,
    edges_process_and_restore_inpainting_by_iqa,
    subgraphs_process_and_restore_inpainting_by_iqa,
    "process_and_restore_inpainting_by_iqa",
)

# main.process_images_with_different_quality_threshold()
nodes_process_images_with_different_quality_threshold = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "CompressImage": "Compress image",
    "SaveCompressedImage": "Save compressed image",
}
edges_process_images_with_different_quality_threshold = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "CompressImage"),
    ("CompressImage", "SaveCompressedImage"),
]
subgraphs_process_images_with_different_quality_threshold = {
    "cluster_process_images_with_different_quality_threshold": {
        "label": "Process Images with Different Quality and Threshold",
        "nodes": nodes_process_images_with_different_quality_threshold,
    }
}
draw_flowchart(
    "Process Images with Different Quality and Threshold",
    nodes_process_images_with_different_quality_threshold,
    edges_process_images_with_different_quality_threshold,
    subgraphs_process_images_with_different_quality_threshold,
    "process_images_with_different_quality_threshold",
)

# main.process_and_analyze_images_with_different_quality_threshold()
nodes_process_and_analyze_images_with_different_quality_threshold = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "CompressImage": "Compress image",
    "CalculateIQA": "Calculate IQA",
    "SaveResults": "Save results",
}
edges_process_and_analyze_images_with_different_quality_threshold = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "CompressImage"),
    ("CompressImage", "CalculateIQA"),
    ("CalculateIQA", "SaveResults"),
]
subgraphs_process_and_analyze_images_with_different_quality_threshold = {
    "cluster_process_and_analyze_images_with_different_quality_threshold": {
        "label": "Process and Analyze Images with Different Quality and Threshold",
        "nodes": nodes_process_and_analyze_images_with_different_quality_threshold,
    }
}
draw_flowchart(
    "Process and Analyze Images with Different Quality and Threshold",
    nodes_process_and_analyze_images_with_different_quality_threshold,
    edges_process_and_analyze_images_with_different_quality_threshold,
    subgraphs_process_and_analyze_images_with_different_quality_threshold,
    "process_and_analyze_images_with_different_quality_threshold",
)

# main.process_and_restore_images_with_different_quality_threshold_by_iqa()
nodes_process_and_restore_images_with_different_quality_threshold_by_iqa = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "CompressImage": "Compress image",
    "DecompressImage": "Decompress image",
    "RestoreImage": "Restore image",
    "CalculateIQA": "Calculate IQA",
    "SaveResults": "Save results",
}
edges_process_and_restore_images_with_different_quality_threshold_by_iqa = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "CompressImage"),
    ("CompressImage", "DecompressImage"),
    ("DecompressImage", "RestoreImage"),
    ("RestoreImage", "CalculateIQA"),
    ("CalculateIQA", "SaveResults"),
]
subgraphs_process_and_restore_images_with_different_quality_threshold_by_iqa = {
    "cluster_process_and_restore_images_with_different_quality_threshold_by_iqa": {
        "label": "Process and Restore Images with Different Quality and Threshold by IQA",
        "nodes": nodes_process_and_restore_images_with_different_quality_threshold_by_iqa,
    }
}
draw_flowchart(
    "Process and Restore Images with Different Quality and Threshold by IQA",
    nodes_process_and_restore_images_with_different_quality_threshold_by_iqa,
    edges_process_and_restore_images_with_different_quality_threshold_by_iqa,
    subgraphs_process_and_restore_images_with_different_quality_threshold_by_iqa,
    "process_and_restore_images_with_different_quality_threshold_by_iqa",
)

# main.process_and_restored_analyze_images_with_different_quality_threshold()
nodes_process_and_restored_analyze_images_with_different_quality_threshold = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "CompressImage": "Compress image",
    "DecompressImage": "Decompress image",
    "RestoreImage": "Restore image",
    "CalculateIQA": "Calculate IQA",
    "SaveResults": "Save results",
}
edges_process_and_restored_analyze_images_with_different_quality_threshold = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "CompressImage"),
    ("CompressImage", "DecompressImage"),
    ("DecompressImage", "RestoreImage"),
    ("RestoreImage", "CalculateIQA"),
    ("CalculateIQA", "SaveResults"),
]
subgraphs_process_and_restored_analyze_images_with_different_quality_threshold = {
    "cluster_process_and_restored_analyze_images_with_different_quality_threshold": {
        "label": "Process and Restored Analyze Images with Different Quality and Threshold",
        "nodes": nodes_process_and_restored_analyze_images_with_different_quality_threshold,
    }
}
draw_flowchart(
    "Process and Restored Analyze Images with Different Quality and Threshold",
    nodes_process_and_restored_analyze_images_with_different_quality_threshold,
    edges_process_and_restored_analyze_images_with_different_quality_threshold,
    subgraphs_process_and_restored_analyze_images_with_different_quality_threshold,
    "process_and_restored_analyze_images_with_different_quality_threshold",
)

# main.process_images_with_different_quality_threshold_edge_methods()
nodes_process_images_with_different_quality_threshold_edge_methods = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "ApplyEdgeDetection": "Apply edge detection",
    "CompressImage": "Compress image",
    "SaveCompressedImage": "Save compressed image",
}
edges_process_images_with_different_quality_threshold_edge_methods = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "ApplyEdgeDetection"),
    ("ApplyEdgeDetection", "CompressImage"),
    ("CompressImage", "SaveCompressedImage"),
]
subgraphs_process_images_with_different_quality_threshold_edge_methods = {
    "cluster_process_images_with_different_quality_threshold_edge_methods": {
        "label": "Process Images with Different Quality, Threshold, and Edge Methods",
        "nodes": nodes_process_images_with_different_quality_threshold_edge_methods,
    }
}
draw_flowchart(
    "Process Images with Different Quality, Threshold, and Edge Methods",
    nodes_process_images_with_different_quality_threshold_edge_methods,
    edges_process_images_with_different_quality_threshold_edge_methods,
    subgraphs_process_images_with_different_quality_threshold_edge_methods,
    "process_images_with_different_quality_threshold_edge_methods",
)

# main.process_and_analyze_images_with_different_quality_threshold_edge_methods()
nodes_process_and_analyze_images_with_different_quality_threshold_edge_methods = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "ApplyEdgeDetection": "Apply edge detection",
    "CompressImage": "Compress image",
    "CalculateIQA": "Calculate IQA",
    "SaveResults": "Save results",
}
edges_process_and_analyze_images_with_different_quality_threshold_edge_methods = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "ApplyEdgeDetection"),
    ("ApplyEdgeDetection", "CompressImage"),
    ("CompressImage", "CalculateIQA"),
    ("CalculateIQA", "SaveResults"),
]
subgraphs_process_and_analyze_images_with_different_quality_threshold_edge_methods = {
    "cluster_process_and_analyze_images_with_different_quality_threshold_edge_methods": {
        "label": "Process and Analyze Images with Different Quality, Threshold, and Edge Methods",
        "nodes": nodes_process_and_analyze_images_with_different_quality_threshold_edge_methods,
    }
}
draw_flowchart(
    "Process and Analyze Images with Different Quality, Threshold, and Edge Methods",
    nodes_process_and_analyze_images_with_different_quality_threshold_edge_methods,
    edges_process_and_analyze_images_with_different_quality_threshold_edge_methods,
    subgraphs_process_and_analyze_images_with_different_quality_threshold_edge_methods,
    "process_and_analyze_images_with_different_quality_threshold_edge_methods",
)

# main.process_and_restore_images_with_different_quality_threshold_edge_methods()
nodes_process_and_restore_images_with_different_quality_threshold_edge_methods = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "ApplyEdgeDetection": "Apply edge detection",
    "CompressImage": "Compress image",
    "DecompressImage": "Decompress image",
    "RestoreImage": "Restore image",
    "SaveRestoredImage": "Save restored image",
}
edges_process_and_restore_images_with_different_quality_threshold_edge_methods = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "ApplyEdgeDetection"),
    ("ApplyEdgeDetection", "CompressImage"),
    ("CompressImage", "DecompressImage"),
    ("DecompressImage", "RestoreImage"),
    ("RestoreImage", "SaveRestoredImage"),
]
subgraphs_process_and_restore_images_with_different_quality_threshold_edge_methods = {
    "cluster_process_and_restore_images_with_different_quality_threshold_edge_methods": {
        "label": "Process and Restore Images with Different Quality, Threshold, and Edge Methods",
        "nodes": nodes_process_and_restore_images_with_different_quality_threshold_edge_methods,
    }
}
draw_flowchart(
    "Process and Restore Images with Different Quality, Threshold, and Edge Methods",
    nodes_process_and_restore_images_with_different_quality_threshold_edge_methods,
    edges_process_and_restore_images_with_different_quality_threshold_edge_methods,
    subgraphs_process_and_restore_images_with_different_quality_threshold_edge_methods,
    "process_and_restore_images_with_different_quality_threshold_edge_methods",
)

# main.process_mask_images_with_different_parameters()
nodes_process_mask_images_with_different_parameters = {
    "LoadImage": "Load image",
    "SetThreshold": "Set threshold",
    "ApplyEdgeDetection": "Apply edge detection",
    "CreateMask": "Create mask",
    "SaveMask": "Save mask",
}
edges_process_mask_images_with_different_parameters = [
    ("LoadImage", "SetThreshold"),
    ("SetThreshold", "ApplyEdgeDetection"),
    ("ApplyEdgeDetection", "CreateMask"),
    ("CreateMask", "SaveMask"),
]
subgraphs_process_mask_images_with_different_parameters = {
    "cluster_process_mask_images_with_different_parameters": {
        "label": "Process Mask Images with Different Parameters",
        "nodes": nodes_process_mask_images_with_different_parameters,
    }
}
draw_flowchart(
    "Process Mask Images with Different Parameters",
    nodes_process_mask_images_with_different_parameters,
    edges_process_mask_images_with_different_parameters,
    subgraphs_process_mask_images_with_different_parameters,
    "process_mask_images_with_different_parameters",
)

# main.process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path()
nodes_process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "ApplyEdgeDetection": "Apply edge detection",
    "SetInpaintingParameters": "Set inpainting parameters",
    "ProcessImage": "Process image",
    "AnalyzeImage": "Analyze image",
    "SaveResults": "Save results",
}
edges_process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "ApplyEdgeDetection"),
    ("ApplyEdgeDetection", "SetInpaintingParameters"),
    ("SetInpaintingParameters", "ProcessImage"),
    ("ProcessImage", "AnalyzeImage"),
    ("AnalyzeImage", "SaveResults"),
]
subgraphs_process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path = {
    "cluster_process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path": {
        "label": "Process and Analyze Images with Different Quality, Threshold, Edge Methods, and Inpainting Parameters",
        "nodes": nodes_process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path,
    }
}
draw_flowchart(
    "Process and Analyze Images with Different Quality, Threshold, Edge Methods, and Inpainting Parameters",
    nodes_process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path,
    edges_process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path,
    subgraphs_process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path,
    "process_and_analyze_images_with_different_quality_threshold_edge_methods_by_different_path",
)

# main.process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path()
nodes_process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path = {
    "LoadImage": "Load image",
    "SetQuality": "Set quality",
    "SetThreshold": "Set threshold",
    "ApplyEdgeDetection": "Apply edge detection",
    "SetInpaintingParameters": "Set inpainting parameters",
    "ProcessImage": "Process image",
    "DecompressImage": "Decompress image",
    "RestoreImage": "Restore image",
    "SaveRestoredImage": "Save restored image",
}
edges_process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path = [
    ("LoadImage", "SetQuality"),
    ("SetQuality", "SetThreshold"),
    ("SetThreshold", "ApplyEdgeDetection"),
    ("ApplyEdgeDetection", "SetInpaintingParameters"),
    ("SetInpaintingParameters", "ProcessImage"),
    ("ProcessImage", "DecompressImage"),
    ("DecompressImage", "RestoreImage"),
    ("RestoreImage", "SaveRestoredImage"),
]
subgraphs_process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path = {
    "cluster_process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path": {
        "label": "Process and Restore Images with Different Quality, Threshold, Edge Methods, and Inpainting Parameters",
        "nodes": nodes_process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path,
    }
}
draw_flowchart(
    "Process and Restore Images with Different Quality, Threshold, Edge Methods, and Inpainting Parameters",
    nodes_process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path,
    edges_process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path,
    subgraphs_process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path,
    "process_and_restore_images_with_different_quality_threshold_edge_methods_by_different_path",
)

# main.highlight_image_by_mask()
nodes_highlight_image_by_mask = {
    "LoadImage": "Load image",
    "CreateMask": "Create mask",
    "HighlightImage": "Highlight image",
    "SaveMask": "Save mask",
    "SaveHighlightedImage": "Save highlighted image",
}
edges_highlight_image_by_mask = [
    ("LoadImage", "CreateMask"),
    ("CreateMask", "HighlightImage"),
    ("HighlightImage", "SaveMask"),
    ("HighlightImage", "SaveHighlightedImage"),
]
subgraphs_highlight_image_by_mask = {
    "cluster_highlight_image_by_mask": {
        "label": "Highlight Image by Mask Process",
        "nodes": nodes_highlight_image_by_mask,
    }
}
draw_flowchart(
    "Highlight Image by Mask",
    nodes_highlight_image_by_mask,
    edges_highlight_image_by_mask,
    subgraphs_highlight_image_by_mask,
    "highlight_image_by_mask",
)
