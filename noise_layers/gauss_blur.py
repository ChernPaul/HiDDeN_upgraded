import numbers

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
# from torchvision.transforms import functional as F
# from torchvision.transforms _functional_tensor as F_t
from collections.abc import Sequence


# class GaussianBlur(torch.nn.Module):
#     """Blurs image with randomly chosen Gaussian blur.
#     If the image is torch Tensor, it is expected
#     to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.
#
#     Args:
#         kernel_size (int or sequence): Size of the Gaussian kernel.
#         sigma (float or tuple of float (min, max)): Standard deviation to be used for
#             creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
#             of float (min, max), sigma is chosen uniformly at random to lie in the
#             given range.
#
#     Returns:
#         PIL Image or Tensor: Gaussian blurred version of the input image.
#
#     """
#
#     def __init__(self, kernel_size, sigma=(0.1, 2.0)):
#         super().__init__()
#         self.kernel_size = kernel_size
#         for ks in self.kernel_size:
#             if ks <= 0 or ks % 2 == 0:
#                 raise ValueError("Kernel size value should be an odd and positive number.")
#
#         if isinstance(sigma, numbers.Number):
#             if sigma <= 0:
#                 raise ValueError("If sigma is a single number, it must be positive.")
#             sigma = (sigma, sigma)
#
#         elif isinstance(sigma, Sequence) and len(sigma) == 2:
#             if not 0.0 < sigma[0] <= sigma[1]:
#                 raise ValueError("sigma values should be positive and of the form (min, max).")
#         else:
#             raise ValueError("sigma should be a single number or a list/tuple with length 2.")
#
#         self.sigma = sigma
#
#
#     def get_params(sigma_min: float, sigma_max: float) -> float:
#         """Choose sigma for random gaussian blurring.
#
#         Args:
#             sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
#             sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.
#
#         Returns:
#             float: Standard deviation to be passed to calculate kernel for gaussian blurring.
#         """
#         return torch.empty(1).uniform_(sigma_min, sigma_max).item()
#
#
#     def forward(self, img: Tensor) -> Tensor:
#         """
#         Args:
#             img (PIL Image or Tensor): image to be blurred.
#
#         Returns:
#             PIL Image or Tensor: Gaussian blurred image
#         """
#         sigma = self.get_params(self.sigma[0], self.sigma[1])
#         return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
#
#
#     def __repr__(self) -> str:
#         s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})"
#         return s
#
#
#
#     def _setup_size(size, error_msg):
#         if isinstance(size, numbers.Number):
#             return int(size), int(size)
#
#         if isinstance(size, Sequence) and len(size) == 1:
#             return size[0], size[0]
#
#         if len(size) != 2:
#             raise ValueError(error_msg)
#
#         return size
#
#
#     def _check_sequence_input(x, name, req_sizes):
#         msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
#         if not isinstance(x, Sequence):
#             raise TypeError(f"{name} should be a sequence of length {msg}.")
#         if len(x) not in req_sizes:
#             raise ValueError(f"{name} should be a sequence of length {msg}.")


#
# def gaussian_blur(img: Tensor, kernel_size, sigma):
#     if not isinstance(kernel_size, (int, list, tuple)):
#         raise TypeError(f"kernel_size should be int or a sequence of integers. Got {type(kernel_size)}")
#     if isinstance(kernel_size, int):
#         kernel_size = [kernel_size, kernel_size]
#     if len(kernel_size) != 2:
#         raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
#     for ksize in kernel_size:
#         if ksize % 2 == 0 or ksize < 0:
#             raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")
#
#     if sigma is None:
#         sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]
#
#     if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
#         raise TypeError(f"sigma should be either float or sequence of floats. Got {type(sigma)}")
#     if isinstance(sigma, (int, float)):
#         sigma = [float(sigma), float(sigma)]
#     if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
#         sigma = [sigma[0], sigma[0]]
#     if len(sigma) != 2:
#         raise ValueError(f"If sigma is a sequence, its length should be 2. Got {len(sigma)}")
#     for s in sigma:
#         if s <= 0.0:
#             raise ValueError(f"sigma should have positive values. Got {sigma}")
#
#     t_img = img
#
#     output = F_t.gaussian_blur(t_img, kernel_size, sigma)
#
#     if not isinstance(img, torch.Tensor):
#         output = to_pil_image(output, mode=img.mode)
#     return output

class Gauss_blur(nn.Module):
    def __init__(self, kernel_size, sigma_param):
        super(Gauss_blur, self).__init__()
        self.sigma_param = sigma_param
        self.kernel_size = kernel_size

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        return noised_and_cover