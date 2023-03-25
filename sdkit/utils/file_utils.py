import json
import os

import piexif
import piexif.helper
import safetensors.torch
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from io import TextIOWrapper
from typing import Any, Dict, List, Union

def load_tensor_file(path):
    if path.lower().endswith(".safetensors"):
        return safetensors.torch.load_file(path, device="cpu")
    else:
        return torch.load(path, map_location="cpu")


def save_tensor_file(data, path):
    if path.lower().endswith(".safetensors"):
        return safetensors.torch.save_file(data, path, metadata={"format": "pt"})
    else:
        return torch.save(data, path)


def save_images(images: list, dir_path: str, file_name="image", output_format="JPEG", output_quality=75):
    """
    * images: a list of of PIL.Image images to save
    * dir_path: the directory path where the images will be saved
    * file_name: the file name to save. Can be a string or a function.
        if a string, the actual file name will be `{file_name}_{index}`.
        if a function, the callback function will be passed the `index` (int),
          and the returned value will be used as the actual file name. e.g `def fn(i): return 'foo' + i`
    * output_format: 'JPEG', 'PNG', or 'WEBP'
    * output_quality: an integer between 0 and 100, used for JPEG and WEBP
    """
    if dir_path is None:
        return
    os.makedirs(dir_path, exist_ok=True)

    for i, img in enumerate(images):
        actual_file_name = file_name(i) if callable(file_name) else f"{file_name}_{i}"
        path = os.path.join(dir_path, actual_file_name)
        img.save(f"{path}.{output_format.lower()}", quality=output_quality)

def save_dicts(
    entries: List[Dict[str, Any]],
    dir_path: str,
    file_name="data",
    output_format: Union[str, List[str]]="txt",
    file_format="",
):
    """
    * entries: a list of dictionaries
    * dir_path: the directory path where the files will be saved
    * file_name: the file name to save. Can be a string or a function.
        if a string, the actual file name will be `{file_name}_{index}`.
        if a function, the callback function will be passed the `index` (int),
          and the returned value will be used as the actual file name. e.g `def fn(i): return 'foo' + i`
    * output_format: 'txt', 'json', or 'embed'
        if 'embed', the metadata will be embedded in PNG files in tEXt chunks, and as EXIF UserComment for JPEG and WEBP files
    """
    if dir_path is None:
        return
    os.makedirs(dir_path, exist_ok=True)

    output_format = [output_format.lower()] if output_format is str else [f.lower() for f in output_format]
    if len(output_format) == 0:
        return

    def save_text_metadata(metadata: Dict[str, Any], f: TextIOWrapper):
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")

    def save_json_metadata(metadata: Dict[str, Any], f: TextIOWrapper):
        json.dump(metadata, f, indent=2)

    metadata_file_types = {
        'json': save_json_metadata, 
        'txt': save_text_metadata,
    }

    for i, metadata in enumerate(entries):
        actual_file_name = file_name(i) if callable(file_name) else f"{file_name}_{i}"
        path = os.path.join(dir_path, actual_file_name)

        for format in output_format:
            if format == "embed":
                targetImage = Image.open(f"{path}.{file_format.lower()}")
                if file_format.lower() == "png":
                    embedded_metadata = PngInfo()
                    for key, val in metadata.items():
                        embedded_metadata.add_text(key, str(val))
                    targetImage.save(f"{path}.{file_format.lower()}", pnginfo=embedded_metadata)
                else:
                    user_comment = json.dumps(metadata)
                    exif_dict = {
                        "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(user_comment, encoding="unicode")}
                    }
                    exif_bytes = piexif.dump(exif_dict)
                    targetImage.save(f"{path}.{file_format.lower()}", exif=exif_bytes)
            elif format in metadata_file_types:
                with open(f"{path}.{format}", "w", encoding="utf-8") as f:
                    metadata_file_types[format](metadata, f)
