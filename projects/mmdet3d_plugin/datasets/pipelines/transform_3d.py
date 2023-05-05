# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
from scipy.spatial.transform import Rotation as R
import mmcv
from mmdet.datasets.builder import PIPELINES
import copy
import inspect
import torch
import cv2
try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None
from PIL import Image

LARGE_DEPTH = 999.


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results['img']
            ]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ResizeMultiview3D:
    """Resize images & bbox & mask.
    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.
    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        # results['scale'] = (1280, 720)
        img_shapes = []
        pad_shapes = []
        scale_factors = []
        keep_ratios = []
        for i in range(len(results['img'])):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'][i], results['scale'], return_scale=True, backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'][i].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'][i], results['scale'], return_scale=True, backend=self.backend)
            results['img'][i] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            img_shapes.append(img.shape)
            pad_shapes.append(img.shape)
            scale_factors.append(scale_factor)
            keep_ratios.append(self.keep_ratio)
            #rescale the camera intrinsic
            results['intrinsics'][i][0, 0] *= w_scale
            results['intrinsics'][i][0, 2] *= w_scale
            results['intrinsics'][i][1, 1] *= h_scale
            results['intrinsics'][i][1, 2] *= h_scale
        results['img_shape'] = img_shapes
        results['pad_shape'] = pad_shapes
        results['scale_factor'] = scale_factors
        results['keep_ratio'] = keep_ratios

        results['lidar2img'] = [
            results['intrinsics'][i] @ np.linalg.inv(results['extrinsics'][i])
            for i in range(len(results['extrinsics']))
        ]

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, ('scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]

        results["img"] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ np.linalg.inv(results['extrinsics'][i]) for i in range(N)]

        return results

    def _get_rot(self, h):

        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)
        results['rot_angle'] = rot_angle

        self.rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle *= -1
        results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        results['scale_ratio'] = scale_ratio
        self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            # K * E^(-1) * R^(-1) * (R * x)
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            # K * (R * E)^(-1) * (R * x)
            # intuitively, the camera rig is transformed by the same rotation as the boxes
            results['extrinsics'][view] = (rot_mat @ torch.tensor(results["extrinsics"][view]).float()).numpy()

        return

    def scale_xyz(self, results, scale_ratio):
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            # according to similar triangle, the translation part of camera rig should scale proportionally
            # in order to result in the same image
            results["extrinsics"][view][:3, 3] *= scale_ratio

        results['lidar2img'] = [
            results['intrinsics'][i] @ np.linalg.inv(results['extrinsics'][i]) for i in range(num_view)
        ]

        return


@PIPELINES.register_module()
class AlbuMultiview3D:
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(
        self,
        transforms,
        keymap=None,
        update_pad_shape=False,
    ):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [self.albu_builder(transform) for transform in args['transforms']]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        aug_imgs = []
        for i in range(len(results["image"])):
            tmp = dict(image=results["image"][i])
            aug_imgs.append(self.aug(**tmp)["image"])

        results["image"] = aug_imgs
        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(self.saturation_lower, self.saturation_upper)

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipImageFull3D(object):
    """Random resize, Crop and flip the image. When flipped, the camera rig and 3D world are also flipped.
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def _resize_depth_preserve(self, resize_dims, depth_map):
        """
        Adapted from:
            https://github.com/TRI-ML/packnet-sfm_internal/blob/919ab604ae2319e4554d3b588877acfddf877f9c/packnet_sfm/datasets/augmentations.py#L93
        -------------------------------------------------------------------------------------------------------------------
        Resizes depth map preserving all valid depth pixels
        Multiple downsampled points can be assigned to the same pixel.
        Parameters
        ----------
        depth : np.array [h,w]
            Depth map
        shape : tuple (H,W)
            Output shape
        Returns
        -------
        depth : np.array [H,W,1]
            Resized depth map
        """
        new_shape = (resize_dims[1], resize_dims[0])

        h, w = depth_map.shape
        x = depth_map.reshape(-1)
        # Create coordinate grid
        uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
        # Filters valid points
        idx = x > 0
        crd, val = uv[idx], x[idx]
        # Downsamples coordinates
        crd[:, 0] = (crd[:, 0] * (new_shape[0] / h)).astype(np.int32)
        crd[:, 1] = (crd[:, 1] * (new_shape[1] / w)).astype(np.int32)
        # Filters points inside image
        idx = (crd[:, 0] < new_shape[0]) & (crd[:, 1] < new_shape[1])
        crd, val = crd[idx], val[idx]
        # Creates downsampled depth image and assigns points
        resized_depth_map = np.zeros(new_shape, dtype=depth_map.dtype)
        resized_depth_map[crd[:, 0], crd[:, 1]] = val
        return resized_depth_map

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """

        imgs = results["img"]
        depth_maps = results.get("depth_maps", [])
        gt_depth_maps = results.get('gt_depth_maps', [])
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, _ = self._sample_augmentation()
        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            depth_map = depth_maps[i] if len(depth_maps) > 0 else None
            gt_depth_map = gt_depth_maps[i] if len(gt_depth_maps) > 0 and i < len(gt_depth_maps) else None

            # resize
            img = img.resize(resize_dims)
            factor_x = img.width / imgs[i].shape[1]
            factor_y = img.height / imgs[i].shape[0]
            results['intrinsics'][i][0] *= factor_x
            results['intrinsics'][i][1] *= factor_y
            if depth_map is not None:
                if depth_map.shape[0] < resize_dims[1]:
                    depth_map = cv2.resize(depth_map, resize_dims, interpolation=cv2.INTER_NEAREST)
                else:
                    depth_map = self._resize_depth_preserve(resize_dims, depth_map)
            if gt_depth_map is not None:
                if gt_depth_map.shape[0] < resize_dims[1]:
                    gt_depth_map = cv2.resize(gt_depth_map, resize_dims, interpolation=cv2.INTER_NEAREST)
                else:
                    gt_depth_map = self._resize_depth_preserve(resize_dims, gt_depth_map)

            # crop
            img = img.crop(crop)
            results['intrinsics'][i][0, 2] -= crop[0]
            results['intrinsics'][i][1, 2] -= crop[1]
            if depth_map is not None:
                depth_map = depth_map[crop[1]:crop[3], crop[0]:crop[2]]
            if gt_depth_map is not None:
                gt_depth_map = gt_depth_map[crop[1]:crop[3], crop[0]:crop[2]]

            # horizontal flip
            if flip:
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
                results['intrinsics'][i][0, 2] = img.width - results['intrinsics'][i][0, 2]

                # augment extrinsics
                rot_mat, tvec = results['extrinsics'][i][:3, :3], results['extrinsics'][i][:3, 3]
                rot_x, rot_y, rot_z = R.from_matrix(rot_mat).as_euler('xyz')
                rot_mat = R.from_euler('xyz', [rot_x, -rot_y, -rot_z]).as_matrix()
                tvec[0] *= -1
                results['extrinsics'][i][:3, :3] = rot_mat
                results['extrinsics'][i][:3, 3] = tvec
                if depth_map is not None:
                    depth_map = np.flip(depth_map, axis=1).copy()
                if gt_depth_map is not None:
                    gt_depth_map = np.flip(gt_depth_map, axis=1).copy()

            new_imgs.append(np.array(img).astype(np.float32))
            if depth_map is not None:
                # NOTE: PIL's crop automatically takes care of out-of-range cropping by padding
                if depth_map.shape != new_imgs[-1].shape[:2]:
                    depth_map_padded = np.ones_like(new_imgs[-1][:, :, 0]) * LARGE_DEPTH
                    h, w = depth_map.shape
                    depth_map_padded[:h, :w] = depth_map
                    depth_maps[i] = depth_map_padded
                else:
                    depth_maps[i] = depth_map
            if gt_depth_map is not None:
                # NOTE: PIL's crop automatically takes care of out-of-range cropping by padding
                if gt_depth_map.shape != new_imgs[-1].shape[:2]:
                    gt_depth_map_padded = np.ones_like(new_imgs[-1][:, :, 0]) * LARGE_DEPTH
                    h, w = gt_depth_map.shape
                    gt_depth_map_padded[:h, :w] = gt_depth_map
                    gt_depth_maps[i] = gt_depth_map_padded
                else:
                    gt_depth_maps[i] = gt_depth_map

        results["img"] = new_imgs
        if len(depth_maps) > 0:
            results["depth_maps"] = depth_maps
        if len(gt_depth_maps) > 0:
            results['gt_depth_maps'] = gt_depth_maps
        results['lidar2img'] = [results['intrinsics'][i] @ np.linalg.inv(results['extrinsics'][i]) for i in range(N)]

        # augment boxes
        if flip:
            # flip x component and yaw
            results['gt_bboxes_3d'].flip('vertical')

        return results

    def _get_rot(self, h):

        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class ProjectPointCloud(object):
    """Random resize, Crop and flip the image. When flipped, the camera rig and 3D world are also flipped.
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, num_views=6, keyframe_only=False):
        self.num_views = num_views
        self.keyframe_only = keyframe_only

    def __call__(self, results):
        points = results['points'].tensor.clone()
        assert points.shape[-1] == 4
        points[..., -1] = 1.0
        points = points.repeat(len(results['intrinsics']), 1, 1).transpose(1, 2)  # (N, 4, L)

        lidar2cam = torch.stack(
            [torch.as_tensor(np.linalg.inv(extr), dtype=points.dtype) for extr in results['extrinsics']])
        points = torch.bmm(lidar2cam, points)

        intrinsics = torch.stack([torch.as_tensor(intr, dtype=points.dtype) for intr in results['intrinsics']])
        H, W = results['pad_shape'][:2]
        uv = torch.bmm(intrinsics, points)[:, :3, :]
        uv = uv / (uv[:, 2:3, :] + 1e-6)

        gt_depth_maps = []
        num_views = self.num_views if self.keyframe_only else len(results['intrinsics'])
        for i in range(num_views):
            # consider only points in front of camera
            valid_mask = points[i, 2, :] > 0
            valid_pts = points[i, :3, valid_mask]
            uv_i = uv[i, :2, valid_mask]

            # consider points that fall in the camera canvas
            valid_mask = (uv_i[0] > 0) & (uv_i[0] < W) & (uv_i[1] > 0) & (uv_i[1] < H)
            uv_i = uv_i[:, valid_mask].to(torch.long)
            valid_pts = valid_pts[:, valid_mask]

            gt_depth_map = torch.zeros((H, W), dtype=points.dtype)
            gt_depth_map[uv_i[1], uv_i[0]] = valid_pts[2]
            gt_depth_maps.append(gt_depth_map.numpy())

        results['gt_depth_maps'] = gt_depth_maps
        return results


@PIPELINES.register_module()
class ComputeMultiviewTargets(object):
    """
    Prepare the decoding view frames, and if given gt, compute the multiview targets in view frames.
    
    """

    def __init__(self,
                 local_frame=True,
                 visible_only=True,
                 num_views=2,
                 keyframe_only=True,
                 virtual_xyz_ranges=[[-0.6, 0.6], [-1.0, 1.0], [-0.3, 0.]],
                 virtual_yaw_range=[-1, 1],
                 use_virtual=False,
                 simulate_cam=False) -> None:
        self.local_frame = local_frame
        self.visible_only = visible_only
        self.num_views = num_views
        self.keyframe_only = keyframe_only
        self.virtual_xyz_ranges = np.array(virtual_xyz_ranges, dtype=np.float32)
        self.virtual_yaw_range = virtual_yaw_range
        self.use_virtual = use_virtual
        self.simulate_cam = simulate_cam

    def sample_virtual_cam(self):
        tvec = np.random.uniform(size=(3, ))
        tvec = tvec * (self.virtual_xyz_ranges[:, 1] - self.virtual_xyz_ranges[:, 0]) + self.virtual_xyz_ranges[:, 0]
        yaw = np.pi * np.random.uniform(self.virtual_yaw_range[0], self.virtual_yaw_range[1])
        cos, sin = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos, 0, -sin], [sin, 0, cos], [0, -1, 0]])

        extrinsic = np.concatenate([rot, tvec[:, None]], axis=-1)
        extrinsic = np.concatenate([extrinsic, np.array([[0., 0., 0., 1.0]])], axis=0).astype(np.float32)

        return extrinsic

    def __call__(self, results) -> None:
        """
        M boxes, N views, compute:
        center targets: M x (N * 3)
        visibility mask: M x N
        """
        num_targets = results['gt_bboxes_3d'].tensor.shape[0] if 'gt_bboxes_3d' in results else 0
        num_views = self.num_views if self.keyframe_only or self.use_virtual else len(results['intrinsics'])

        targets_all, masks_all, dec_extrinsics = [], [], []
        for i in range(num_views):
            extr = self.sample_virtual_cam() if self.use_virtual else results['extrinsics'][i].copy()
            # change axes to x-right, y-front, z-up for convenience
            if not self.simulate_cam:
                extr[:3, :3] = np.matmul(extr[:3, :3], np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]))
                yaw = (np.arctan2(extr[1, 0], extr[0, 0]) + np.arctan2(-extr[0, 1], extr[1, 1])) / 2
            else:
                yaw = (np.arctan2(extr[1, 0], extr[0, 0]) + np.arctan2(-extr[0, 2], extr[1, 2])) / 2
            dec_extrinsics.append(extr.astype(np.float32))

            if 'gt_bboxes_3d' not in results:
                continue

            # get targets
            centers = np.concatenate([results['gt_bboxes_3d'].gravity_center, np.ones((num_targets, 1))], axis=-1)
            velos = np.concatenate([results['gt_bboxes_3d'].tensor[:, -2:], np.zeros((num_targets, 1))], axis=-1)
            wlh = results['gt_bboxes_3d'].tensor[:, 3:6]
            yaws = results['gt_bboxes_3d'].tensor[:, 6:7]
            if self.local_frame:
                centers = np.matmul(np.linalg.inv(extr)[None, ...], centers[..., None])[:, :3, 0]
                velos = np.matmul(extr[:3, :3].transpose()[None, ...], velos[..., None])[:, :2, 0]
                # NOTE: the definition of box's yaw in mmdet3d version in this docker is negative to right-hand convention, so we use "+" here
                yaws = yaws + yaw
            else:
                centers = centers[:, :3]
                velos = velos[:, :2]
            targets = np.concatenate([centers, wlh, yaws, velos], axis=-1)
            targets = targets.astype(extr.dtype)

            if self.visible_only:
                # get visibility mask
                corners = np.concatenate([results['gt_bboxes_3d'].corners, np.ones((num_targets, 8, 1))], axis=-1)
                corners = np.matmul(np.linalg.inv(extr)[None, None, ...], corners[..., None])

                # criterion #1: in front of camera
                visible = corners[..., 2, 0] > 0

                # criterion #2: inside image canvas
                intr = results['intrinsics'][i]
                H, W = results['img'][i].shape[:2]
                corners_2d = np.matmul(intr[None, None, ...], corners)[..., :3, 0]
                corners_2d = corners_2d / (corners_2d[..., 2:] + 1e-6)
                inside = (corners_2d[..., 0] > 0) & (corners_2d[..., 0] < W) & (corners_2d[..., 1] > 0) & (
                    corners_2d[..., 1] < H)

                masks = (visible & inside).any(axis=1).astype(extr.dtype)
            else:
                masks = np.ones((num_targets), dtype=extr.dtype)

            targets_all.append(targets)
            masks_all.append(masks)

        if num_targets > 0:
            targets_all = np.stack(targets_all).transpose(1, 0, 2).reshape(num_targets, -1)
            masks_all = np.stack(masks_all).transpose(1, 0)
        else:
            targets_all = np.zeros((num_targets, num_views * 3), dtype=extr.dtype)
            masks_all = np.zeros((num_targets), dtype=extr.dtype)

        if 'gt_bboxes_3d' in results:
            results['gt_bboxes_3d'].mtv_targets = torch.as_tensor(targets_all)
            results['gt_bboxes_3d'].mtv_visibility = torch.as_tensor(masks_all)
        results['dec_extrinsics'] = dec_extrinsics

        return results
