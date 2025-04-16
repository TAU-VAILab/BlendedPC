import ailia
import os
import urllib.request
import ssl
import shutil

CATEGORIES = {
    "chair": {
        "back": [0],
        "seat": [1],
        "leg": [2],
        "arm": [3],
    },
    "table": {
        "top": [0],
        "leg": [1],
        "support": [2],
    },
    "lamp": {
        "base": [0],
        "shade": [1],
        "bulb": [2],
        "tube": [3],
    },
    "airplane": {
        "body": [0],
        "wing": [1],
        "tail": [2],
        "engine": [3],
    },
    "sofa": {
        "back": [0],
        "seat": [1],
        "leg": [2],
        "arm": [3],
    },
    "knife": {
        "blade": [0],
        "handle": [1],
    },
    "cap": {
        "brim": [0],
        "crown": [1],
    },
    "skateboard": {
        "wheel": [0],
        "deck": [1],
        "truck": [2],
    },
    "mug": {
        "handle": [0],
        "body": [1],
    },
    "pistol": {
        "barrel": [0],
        "handle": [1],
        "trigger": [2],
    },
    "bag": {
        "strap": [0],
        "body": [1],
    },
    "guitar": {
        "neck": [1],
        "body": [2],
        "head": [0],
    }
}


# source: https://github.com/axinc-ai/ailia-models/blob/master/util/model_utils.py

def progress_print(block_count, block_size, total_size):
    """
    Callback function to display the progress
    (ref: https://qiita.com/jesus_isao/items/ffa63778e7d3952537db)

    Parameters
    ----------
    block_count:
    block_size:
    total_size:
    """
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        # Bigger than 100 does not look good, so...
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = '=' * bar_num
    if bar_num != max_bar:
        progress_element += '>'
    bar_fill = ' '  # fill the blanks
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(f'[{bar} {percentage:.2f}% ( {total_size_kb:.0f}KB )]', end='\r')


def urlretrieve(remote_path, weight_path, progress_print):
    temp_path = weight_path + ".tmp"
    try:
        # raise ssl.SSLError # test
        urllib.request.urlretrieve(
            remote_path,
            temp_path,
            progress_print,
        )
    except ssl.SSLError as e:
        remote_path = remote_path.replace("https", "http")
        urllib.request.urlretrieve(
            remote_path,
            temp_path,
            progress_print,
        )
    shutil.move(temp_path, weight_path)


def check_and_download_models(weight_path, model_path, remote_path):
    """
    Check if the onnx file and prototxt file exists,
    and if necessary, download the files to the given path.

    Parameters
    ----------
    weight_path: string
        The path of onnx file.
    model_path: string
        The path of prototxt file for ailia.
    remote_path: string
        The url where the onnx file and prototxt file are saved.
        ex. "https://storage.googleapis.com/ailia-models/mobilenetv2/"
    """

    if not os.path.exists(weight_path):
        urlretrieve(
            remote_path + os.path.basename(weight_path),
            weight_path,
            progress_print,
        )
    if model_path != None and not os.path.exists(model_path):
        urlretrieve(
            remote_path + os.path.basename(model_path),
            model_path,
            progress_print,
        )


def init_net_seg(shape_category: str):
    """
    Initialize the net_seg for a given shape category.
    """
    assert shape_category in CATEGORIES, f"Invalid shape category: {shape_category}"
    weight_path = f'{shape_category}_100.onnx'
    model_path = f'{shape_category}_100.onnx.prototxt'
    check_and_download_models(
        weight_path, model_path, 'https://storage.googleapis.com/ailia-models/pointnet_pytorch/')
    return ailia.Net(model_path, weight_path)
