import os

import numpy as np
import yaml
from dotmap import DotMap


def get_config_from_yml(yml_file):
    """
    Get the config from a yml file
    :param yml_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(yml_file, "r") as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(yml_file):
    config, _ = get_config_from_yml(yml_file)

    # set data info
    config.data.img_shape = (config.data.img_size, config.data.img_size, config.data.img_channels)
    exp_dir = os.path.join(config.exp.experiment_dir, config.exp.name)
    config.exp.log_dir = os.path.join(exp_dir, "logs")
    config.exp.plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(config.exp.log_dir, exist_ok=True)
    os.makedirs(config.exp.plots_dir, exist_ok=True)
    return config


def scalar_to_onehot(arr, n_class):
    length = arr.shape[0]
    one_hots = np.zeros((length, n_class))
    one_hots[np.arange(length), arr] = 1
    return one_hots


def normalize_img(img):
    """ Normalize image from 0 ~ 255 to -1 ~ 1
    Return: np array with dtype float32
    """
    return img.astype(np.float32) / 127.5 - 1

def rotate(imgs, max_val=1):
    """
    :param imgs: List of images drawn using draw_ch functions (PIL Image)
    :param max_val: Boolean to determine whether to use max value
    :param verbose: Boolean to determine whether to print information
    :return: List of PIL image that is rotated
    """
    imgs_new = []
    angle = -max_val if use_max_val else random.uniform(-max_val, max_val)

    rows, cols = imgs[0].shape[:2]
    center = (rows / 2, cols / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    imgs_new = [
        cv2.warpAffine(
            img, rotation_matrix, (cols, rows), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        for img in imgs
    ]
    if verbose:
        print("Rotated by:", angle)
    return imgs_new

