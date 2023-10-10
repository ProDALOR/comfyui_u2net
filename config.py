import os

from comfy.model_management import get_torch_device
import folder_paths

from .u2net import U2NET

u2net_segmentation_name = "u2net"
category = u2net_segmentation_name

u2net_segmentation_models_path = os.path.join(folder_paths.models_dir, u2net_segmentation_name)

if not os.path.isdir(u2net_segmentation_models_path):
    os.mkdir(u2net_segmentation_models_path)

folder_paths.add_model_folder_path(u2net_segmentation_name, u2net_segmentation_models_path)