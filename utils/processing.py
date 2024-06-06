'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import numbers
import random
from io import BytesIO
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


def make_processing(opt):
    # make precessing transform
    # input: an argparse.Namespace
    # output: a torchvision.transforms.Compose
    #

    opt = parse_arguments(opt)
    transforms_list = list()  # list of transforms

    transforms_pre = make_pre(opt)  # make pre-data-augmentation transforms
    if transforms_pre is not None:
        transforms_list.append(transforms_pre)

    transforms_aug = make_aug(opt)  # make data-augmentation transforms
    if transforms_aug is not None:
        idx_aug = len(transforms_list)
        transforms_list.append(transforms_aug)
    else:
        idx_aug = -1

    transforms_post = make_post(opt)  # make post-data-augmentation transforms
    if transforms_post is not None:
        transforms_list.append(transforms_post)

    transforms_list.append(make_normalize(opt.norm_type))  # make normalization

    if (hasattr(opt, "num_views")) and (abs(opt.num_views) > 0):
        print("num_view:", opt.num_views)
        t = transforms.Compose(transforms_list)
        # make multiviews for Self-supervised learning (SSL)
        t = MultiView([t for _ in range(abs(opt.num_views))])
    else:
        t = transforms.Compose(transforms_list)

    return t


def add_processing_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # ICASSP2023: --cropSize 96 --loadSize -1 --resizeSize -1 --norm_type resnet --resize_prob 0.2 --jitter_prob 0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --hpf_prob 0.0 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100 --resize_size 256 --resize_ratio 0.75
    # ICME2021  : --cropSize 96 --loadSize -1 --resizeSize -1 --norm_type resnet --resize_prob 0.0 --jitter_prob 0.0 --colordist_prob 0.0 --cutout_prob 0.0 --noise_prob 0.0 --blur_prob 0.5 --cmp_prob 0.5 --rot90_prob 1.0 --hpf_prob 0.0 --blur_sig 0.0,3.0 --cmp_method cv2,pil --cmp_qual 30,100
    #

    parser.add_argument(
        "--resizeSize",
        type=int,
        default=-1,
        help="scale images to this size post augumentation",
    )
    parser.add_argument(
        "--loadSize",
        type=int,
        default=-1,
        help="scale images to this size pre augumentation",
    )
    parser.add_argument(
        "--cropSize",
        type=int,
        default=-1,
        help="crop images to this size post augumentation",
    )
    parser.add_argument("--no_random_crop", action="store_true")

    # data-augmentation probabilities
    parser.add_argument("--resize_prob", type=float, default=0.0)
    parser.add_argument("--jitter_prob", type=float, default=0.0)
    parser.add_argument("--colordist_prob", type=float, default=0.0)
    parser.add_argument("--cutout_prob", type=float, default=0.0)
    parser.add_argument("--noise_prob", type=float, default=0.0)
    parser.add_argument("--blur_prob", type=float, default=0.0)
    parser.add_argument("--cmp_prob", type=float, default=0.0)
    parser.add_argument("--rot90_prob", type=float, default=1.0)
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--hpf_prob", type=float, default=0.0)

    # data-augmentation parameters
    parser.add_argument("--rz_interp", default="bilinear")
    parser.add_argument("--blur_sig", default="0.5")
    parser.add_argument("--cmp_method", default="cv2")
    parser.add_argument("--cmp_qual", default="75")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--resize_ratio", type=float, default=1.0)

    # other
    parser.add_argument("--norm_type", type=str, default="resnet")  # normalization type
    # multi views for Self-supervised learning (SSL)
    parser.add_argument("--num_views", type=int, default=0)

    return parser


def parse_arguments(opt):
    if not isinstance(opt.rz_interp, list):
        opt.rz_interp = list(opt.rz_interp.split(","))
    if not isinstance(opt.blur_sig, list):
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")]
    if not isinstance(opt.cmp_method, list):
        opt.cmp_method = list(opt.cmp_method.split(","))
    if not isinstance(opt.cmp_qual, list):
        opt.cmp_qual = [int(s) for s in opt.cmp_qual.split(",")]
        if len(opt.cmp_qual) == 2:
            opt.cmp_qual = list(range(opt.cmp_qual[0], opt.cmp_qual[1] + 1))
        elif len(opt.cmp_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --cmp_qual.")
    return opt


rz_dict = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}


def make_pre(opt):
    transforms_list = list()
    if opt.loadSize > 0:
        print("\nUsing Pre Resizing\n")
        transforms_list.append(
            transforms.Lambda(
                lambda img: TF.resize(
                    img,
                    opt.loadSize,
                    interpolation=rz_dict[sample_discrete(opt.rz_interp)],
                )
            )
        )
        transforms_list.append(
            CenterCropPad(opt.loadSize, pad_if_needed=True, padding_mode="symmetric")
        )

    if len(transforms_list) == 0:
        return None
    else:
        return transforms.Compose(transforms_list)


def make_post(opt):
    transforms_list = list()
    if opt.resizeSize > 0:
        print("\nUsing Post Resizing\n")
        transforms_list.append(
            transforms.Resize(
                opt.resizeSize, interpolation=transforms.InterpolationMode.BICUBIC
            )
        )
        transforms_list.append(transforms.CenterCrop((opt.resizeSize, opt.resizeSize)))

    if opt.cropSize > 0:
        if not opt.no_random_crop:
            print("\nUsing Post Random Crop\n")
            transforms_list.append(
                transforms.RandomCrop(
                    opt.cropSize, pad_if_needed=True, padding_mode="symmetric"
                )
            )
        else:
            print("\nUsing Post Central Crop\n")
            transforms_list.append(
                CenterCropPad(
                    opt.cropSize, pad_if_needed=True, padding_mode="symmetric"
                )
            )

    if len(transforms_list) == 0:
        return None
    else:
        return transforms.Compose(transforms_list)


def make_aug(opt):
    # AUG
    transforms_list_aug = list()

    if (opt.resize_size > 0) and (opt.resize_prob > 0):  # opt.resized_ratio
        transforms_list_aug.append(
            transforms.RandomApply(
                [
                    transforms.RandomResizedCrop(
                        size=opt.resize_size,
                        scale=(0.08, 1.0),
                        ratio=(opt.resize_ratio, 1.0 / opt.resize_ratio),
                        interpolation=rz_dict[sample_discrete(opt.rz_interp)],
                    )
                ],
                opt.resize_prob,
            )
        )

    if opt.jitter_prob > 0:
        transforms_list_aug.append(
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=opt.jitter_prob
            )
        )

    if opt.colordist_prob > 0:
        transforms_list_aug.append(transforms.RandomGrayscale(p=opt.colordist_prob))

    if opt.cutout_prob > 0:
        transforms_list_aug.append(create_cutout_transforms(opt.cutout_prob))

    if opt.noise_prob > 0:
        transforms_list_aug.append(create_noise_transforms(opt.noise_prob))

    if opt.blur_prob > 0:
        transforms_list_aug.append(
            transforms.Lambda(
                lambda img: data_augment_blur(img, opt.blur_prob, opt.blur_sig)
            )
        )

    if opt.cmp_prob > 0:
        transforms_list_aug.append(
            transforms.Lambda(
                lambda img: data_augment_cmp(
                    img, opt.cmp_prob, opt.cmp_method, opt.cmp_qual
                )
            )
        )

    if opt.rot90_prob > 0:
        transforms_list_aug.append(
            transforms.Lambda(lambda img: data_augment_rot90(img, opt.rot90_prob))
        )

    if opt.hpf_prob > 0:
        transforms_list_aug.append(transforms.ToTensor())
        transforms_list_aug.append(
            transforms.Lambda(
                lambda img: data_augment_hpf(img, opt.hpf_prob, opt.blur_sig)
            )
        )

    if not opt.no_flip:
        transforms_list_aug.append(transforms.RandomHorizontalFlip())

    if len(transforms_list_aug) > 0:
        return transforms.Compose(transforms_list_aug)
    else:
        return None


def make_normalize(norm_type):
    transforms_list = list()

    if norm_type == "resnet":
        print("normalize RESNET")

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    elif norm_type == "clip":
        print("normalize CLIP")
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        )
    elif norm_type == "xception":
        print("normalize -1,1")

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
    elif norm_type == "spec":
        print("normalize SPEC")

        transforms_list.append(normalization_fft)
        transforms_list.append(transforms.ToTensor())

    elif norm_type == "fft2":
        print("normalize Energy")

        transforms_list.append(pic2imgn)
        transforms_list.append(normalization_fft2)
        transforms_list.append(imgn2torch)

    elif norm_type == "residue3":
        print("normalize Residue3")

        transforms_list.append(normalization_residue3)
    elif norm_type == "npr":
        print("normalize NPR")

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        from torch.nn.functional import interpolate
        transforms_list.append(
            lambda x: x[..., :(x.shape[-2]//2*2), :(x.shape[-1]//2*2)]
        )
        transforms_list.append(
            lambda x: (x - interpolate(x[None,...,::2,::2], scale_factor=2.0, mode='nearest', recompute_scale_factor=True)[0])*2.0/3.0
        )
    elif norm_type == "cooc":
        print("normalize COOC")

        transforms_list.append(normalization_cooc)
    else:
        assert False

    return transforms.Compose(transforms_list)


class MultiView:
    def __init__(self, trasfroms_list):
        self.trasfroms_list = trasfroms_list
        print("num_view:", len(self.trasfroms_list))

    def __call__(self, x):
        return torch.stack([fun(x) for fun in self.trasfroms_list], 0)


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def data_augment_blur(img, p, blur_sig):
    from scipy.ndimage.filters import gaussian_filter
    if random.random() < p:
        img = np.array(img)
        sig = sample_continuous(blur_sig)
        gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sig)
        gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sig)
        gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sig)
        img = Image.fromarray(img)

    return img


def data_augment_hpf(img, p, blur_sig):
    assert isinstance(img, torch.Tensor)
    if random.random() < p:
        sig = 0.4 + sample_continuous(blur_sig)
        kernel_size = int(7 * sig)
        kernel_size = kernel_size + (kernel_size + 1) % 2
        img = img - TF.gaussian_blur(img, kernel_size=kernel_size, sigma=sig)
        img = img + torch.from_numpy(np.asarray([[[0.485]], [[0.456]], [[0.406]]]))
    return img.float()


def data_augment_cmp(img, p, cmp_method, cmp_qual):
    if random.random() < p:
        img = np.array(img)
        method = sample_discrete(cmp_method)
        qual = sample_discrete(cmp_qual)
        img = cmp_from_key(img, qual, method)
        img = Image.fromarray(img)

    return img


def data_augment_rot90(img, p):
    if random.random() < p:
        angle = sample_discrete([0, 90, 180, 270])
        img = img.rotate(angle, expand=True)

    return img


def data_augment_D4(img, p):
    if random.random() < p:
        angle = sample_discrete([0, 90, 180, 270])
        sim = sample_discrete([0, 1])
        img = img.rotate(angle, expand=True)
        if sim == 1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img



def create_noise_transforms(p, var_limit=(10.0, 50.0)):
    from albumentations.augmentations.transforms import GaussNoise

    aug = GaussNoise(var_limit=var_limit, always_apply=False, p=p)
    return transforms.Lambda(
        lambda img: Image.fromarray(aug(image=np.array(img))["image"])
    )


def create_cutout_transforms(p):
    try:
        from albumentations.augmentations.dropout.cutout import Cutout
    except:
        from albumentations.augmentations.transforms import Cutout
    aug = Cutout(
        num_holes=1,
        max_h_size=48,
        max_w_size=48,
        fill_value=128,
        always_apply=False,
        p=p,
    )
    return transforms.Lambda(
        lambda img: Image.fromarray(aug(image=np.array(img))["image"])
    )


def cv2_webp(img, compress_val):
    import cv2
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), compress_val]
    result, encimg = cv2.imencode(".webp", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_webp(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="webp", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


def cv2_jpg(img, compress_val):
    import cv2
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


# NOTE: 'cv2' and 'pil' have been left here for legacy reasons
cmp_dict = {
    "cv2": cv2_jpg,
    "cv2_jpg": cv2_jpg,
    "cv2_webp": cv2_webp,
    "pil": pil_jpg,
    "pil_jpg": pil_jpg,
    "pil_webp": pil_webp,
}


def cmp_from_key(img, compress_val, key):
    return cmp_dict[key](img, compress_val)


def pic2imgn(pic):
    from copy import deepcopy

    img = np.float32(deepcopy(np.asarray(pic))) / 256.0
    return img


def imgn2torch(img):
    return torch.from_numpy(img).permute(2, 0, 1).float().contiguous()


def normalization_fft2(img, normalize=512.0):
    img = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))
    img = np.square(np.abs(img)) / normalize
    return img


def normalization_fft(pic):
    from copy import deepcopy

    im = np.float32(deepcopy(np.asarray(pic))) / 255.0

    for i in range(im.shape[2]):
        img = im[:, :, i]
        fft_img = np.fft.fft2(img)
        fft_img = np.log(np.abs(fft_img) + 1e-3)
        fft_min = np.percentile(fft_img, 5)
        fft_max = np.percentile(fft_img, 95)
        if (fft_max - fft_min) <= 0:
            print("ma cosa...")
            fft_img = (fft_img - fft_min) / ((fft_max - fft_min) + np.finfo(float).eps)
        else:
            fft_img = (fft_img - fft_min) / (fft_max - fft_min)
        fft_img = (fft_img - 0.5) * 2
        fft_img[fft_img < -1] = -1
        fft_img[fft_img > 1] = 1
        im[:, :, i] = fft_img

    return im


def normalization_residue3(pic, flag_tanh=False):
    from copy import deepcopy

    x = np.float32(deepcopy(np.asarray(pic))) / 32
    wV = (
        -1 * x[1:-3, 2:-2, :]
        + 3 * x[2:-2, 2:-2, :]
        - 3 * x[3:-1, 2:-2, :]
        + 1 * x[4:, 2:-2, :]
    )
    wH = (
        -1 * x[2:-2, 1:-3, :]
        + 3 * x[2:-2, 2:-2, :]
        - 3 * x[2:-2, 3:-1, :]
        + 1 * x[2:-2, 4:, :]
    )
    ress = np.concatenate((wV, wH), -1)
    if flag_tanh:
        ress = np.tanh(ress)

    ress = torch.from_numpy(ress).permute(2, 0, 1).contiguous()

    return ress


def normalization_cooc(pic):
    from copy import deepcopy

    x = deepcopy(np.asarray(pic))
    y = x[1:, 1:, :]
    x = x[:-1, :-1, :]
    bins = np.arange(257)
    H = np.stack(
        [
            np.histogram2d(
                x[:, :, i].flatten(), y[:, :, i].flatten(), bins, density=True
            )[0]
            for i in range(x.shape[2])
        ],
        0,
    )
    H = torch.from_numpy(H)
    return H


class CenterCropPad:
    def __init__(
        self, siz, pad_if_needed=False, padding_fill=0, padding_mode="constant"
    ):
        if isinstance(siz, numbers.Number):
            siz = (int(siz), int(siz))
        self.siz = siz
        self.pad_if_needed = pad_if_needed
        self.padding_fill = padding_fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        crop_height, crop_width = self.siz
        image_width, image_height = img.size[1], img.size[0]
        crop_top = (image_height - crop_height) // 2
        crop_left = (image_width - crop_width) // 2
        if crop_top < 0:
            if self.pad_if_needed:
                img = TF.pad(
                    img,
                    (0, -crop_top, 0, crop_height - image_height + crop_top),
                    fill=self.padding_fill,
                    padding_mode=self.padding_mode,
                )
            else:
                crop_height = image_height
            crop_top = 0
        if crop_left < 0:
            if self.pad_if_needed:
                img = TF.pad(
                    img,
                    (-crop_left, 0, crop_width - image_width + crop_left, 0),
                    fill=self.padding_fill,
                    padding_mode=self.padding_mode,
                )
            else:
                crop_width = image_width
            crop_left = 0
        return img.crop(
            (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
        )
