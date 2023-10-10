import os
from typing import Iterable, Tuple
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

from .u2net import U2NET
from .config import category, u2net_segmentation_models_path, get_torch_device

class NormalizeImage:
    def __init__(self, mean: float, std: float):
        self.normalize_1 = transforms.Normalize(mean, std)
        self.normalize_3 = transforms.Normalize([mean] * 3, [std] * 3)
        self.normalize_18 = transforms.Normalize([mean] * 18, [std] * 18)

    def __call__(self, image_tensor: torch.Tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)
        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)
        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)
        else:
            return image_tensor

### To remove 
class U2NetNormalizeImage:

    def __call__(self, image_tensor: torch.Tensor):
        image = image_tensor.cpu().numpy()

        tmpImg = np.zeros((image.shape[0], image.shape[1],image.shape[2]))
        image = image/np.max(image)

        if image.shape[0]==1:
            tmpImg[0,:,:] = (image[0,:,:]-0.485)/0.229
            tmpImg[1,:,:] = (image[1,:,:]-0.485)/0.229
            tmpImg[2,:,:] = (image[2,:,:]-0.485)/0.229
        else:
            tmpImg[0,:,:] = (image[0,:,:]-0.485)/0.229
            tmpImg[1,:,:] = (image[1,:,:]-0.456)/0.224
            tmpImg[2,:,:] = (image[2,:,:]-0.406)/0.225

        return torch.from_numpy(tmpImg)
###

class U2NetLoader:

    file_formats = [".pth", ".ckpt"]
    replace_prefix = [
        ("module.", ""),
        # ("net.", ""),
        # ("gt_encoder.", "")
    ]

    def __init__(self):
        self.loaded_model = None
        self.loaded_model_name: str = ""
        self.loaded_ch = (0, 0)

    def _check_file_format(file: str) -> bool:
        for f_format in U2NetLoader.file_formats:
            if file.endswith(f_format):
                return True
        return False


    @classmethod
    def INPUT_TYPES(self):
        return {"required": 
                {
                    "model_name": ([f for f in os.listdir(u2net_segmentation_models_path) if os.path.isfile(os.path.join(u2net_segmentation_models_path, f)) and self._check_file_format(f)], ),
                    "in_ch": ("INT", {"default": 3, "min": 1}),
                    "out_ch": ("INT", {"default": 1, "min": 1}),
                }
        }

    RETURN_TYPES = ("U2NET",)
    RETURN_NAMES = ("u2net",)

    FUNCTION = "load_model"

    CATEGORY = category

    @staticmethod
    def _prepare_keys(key: str) -> str:
        for f, r in U2NetLoader.replace_prefix:
            key = key.replace(f, r)
        return key

    def load_model(self, model_name: str, in_ch: int, out_ch: int):
        if model_name != self.loaded_model_name or (in_ch, out_ch) != self.loaded_ch:

            model_path = os.path.join(u2net_segmentation_models_path, model_name)
            
            model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))

            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                new_state_dict[self._prepare_keys(k)] = v

            net = U2NET(in_ch=in_ch, out_ch=out_ch)

            net.load_state_dict(
                new_state_dict
            )

            net = net.to(get_torch_device())

            self.loaded_model = net.eval()
            self.loaded_model_name = model_name
            self.loaded_ch = (in_ch, out_ch)
        
        return (self.loaded_model, )
    
class U2NetSegmentation:

    @classmethod
    def INPUT_TYPES(self):
        return {"required": 
                {
                    "image": ("IMAGE",),
                    "u2net": ("U2NET",),
                    "resize_width": ("INT", {"default": 320, "min": 1}),
                    "resize_height": ("INT", {"default": 320, "min": 1}),
                    "normalize_mean": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                    "normalize_std": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01})
                }
        }

    RETURN_TYPES = ("U2NET_CH",)
    RETURN_NAMES = ("u2net_ch",)

    FUNCTION = "image_to_ch"

    CATEGORY = category

    def image_to_ch(self, image: Iterable[torch.Tensor], u2net: U2NET, resize_width: int, resize_height: int, normalize_mean: float, normalize_std: float):
        

        for image_ in image:

            np_data = np.clip(255. * image_.squeeze().cpu().numpy(), 0, 255).astype(np.uint8)

            img = Image.fromarray(np_data)

            img_size = img.size
            
            img = img.resize((resize_width, resize_height), resample=Image.BICUBIC)

            image_tensor = transforms.Compose([
                transforms.ToTensor(),
                NormalizeImage(normalize_mean, normalize_std),
                # U2NetNormalizeImage()
            ])(img)

            image_tensor =  torch.unsqueeze(image_tensor, 0).type(torch.FloatTensor).to(get_torch_device())

            res,_,_,_,_,_,_ = u2net(image_tensor)
       
            return ((res, img_size), )


def numpy_to_mask(alpha_mask, img_size):
    if len(alpha_mask.shape) == 3:
        alpha_mask = alpha_mask[0]
    alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
    alpha_mask_img = alpha_mask_img.resize(img_size, resample=Image.BICUBIC)
    return torch.from_numpy(np.array(alpha_mask_img) / 255.0).type(torch.FloatTensor)


class U2NetNormalization:

    @classmethod
    def INPUT_TYPES(self):
        return {"required": 
                {
                    "u2net_ch": ("U2NET_CH",)
                }
        }

    RETURN_TYPES = ("U2NET_CH",)
    RETURN_NAMES = ("u2net_ch",)

    FUNCTION = "run_normalization"

    CATEGORY = category

    def normalization(self, output_tensor: torch.Tensor) -> torch.Tensor:
        return output_tensor

    def run_normalization(self, u2net_ch: Tuple[torch.Tensor, Tuple[int, int]]):

        (res, img_size) = u2net_ch

        return ((self.normalization(res), img_size),)


class U2NetBaseNormalization(U2NetNormalization):

    def normalization(self, output_tensor: torch.Tensor) -> torch.Tensor:
        ma = torch.max(output_tensor)
        mi = torch.min(output_tensor)

        dn = (output_tensor-mi)/(ma-mi)

        return dn
    
class U2NetMaxNormalization(U2NetNormalization):

    def normalization(self, output_tensor: torch.Tensor) -> torch.Tensor:
        return torch.max(F.log_softmax(output_tensor, dim=1), dim=1, keepdim=True)[1]


class U2NetToMask:

    @classmethod
    def INPUT_TYPES(self):
        return {"required": 
                {
                    "u2net_ch": ("U2NET_CH",)
                }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "unet_to_mask"

    CATEGORY = category

    def unet_to_mask(self, u2net_ch: Tuple[torch.Tensor, Tuple[int, int]]):

        (res, img_size) = u2net_ch

        output_arr = np.clip(255. * res.squeeze().cpu().numpy(), 0, 255).astype(np.uint8)

        mask = numpy_to_mask(output_arr, img_size)

        return (mask, )
    
        
class U2NetChToMask:

    @classmethod
    def INPUT_TYPES(self):
        return {"required": 
                {
                    "u2net_ch": ("U2NET_CH",),
                    "ch_num": ("INT", {"default": 0, "min": 0}),
                }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "ch_to_mask"

    CATEGORY = category

    def ch_to_mask(self, u2net_ch: Tuple[torch.Tensor, Tuple[int, int]], ch_num: int):

        (res, img_size) = u2net_ch

        output_arr = res.squeeze().cpu().numpy()

        alpha_mask = (output_arr == ch_num).astype(np.uint8) * 255

        mask = numpy_to_mask(alpha_mask, img_size)

        if ch_num <= 0:
            mask = 1. - mask

        return (mask, )
