from __future__ import division
import torch
import sys
import math
import cv2
import numpy as np
import numbers
import collections
import warnings
import random
from PIL import Image, ImageEnhance
from sklearn.preprocessing import normalize as sklearnNorm


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3 and (img.size(0) in {1, 3, 4})


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim == 3) and (img.shape[2] in {1, 3, 4})


def to_tensor(input):
    """Convert a ``numpy.ndarray`` to tensor.

    Args:
        img (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if _is_numpy_image(input):
        input = torch.from_numpy(input.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(input, torch.ByteTensor):
            return input.float().div(255)
        else:
            return input.float()
    else:
        return torch.from_numpy(input).float()

def to_numpy_image(img):
    """Convert a tensor to Numpy Image.

    Args:
        img (Tensor): Image to be converted to numpy image.

    Returns:
        numpy.ndarray: Image converted to numpy image.
    """
    if not _is_tensor_image(img):
        raise TypeError('img should be tensor image. Got {}.'.format(type(img)))

    if isinstance(img, torch.FloatTensor):
        img = img.mul(255).byte()

    npimg = np.transpose(img.numpy(), (1, 2, 0))
    return npimg


def normalize(input, mean, std, inplace=False):
    """Normalize a image with mean and standard deviation.

    Args:
        input (tensor or numpy.ndarray): image to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        img: Normalized image.
    """
    if isinstance(mean, numbers.Number):
        mean = np.array([mean])
    if isinstance(std, numbers.Number):
        std = np.array([std])

    if _is_tensor_image(input):
        if not inplace:
            ret = input.clone()

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        ret.sub_(mean[:, None, None]).div_(std[:, None, None])
    elif _is_numpy_image(input):
        if not inplace:
            ret = input.copy()

        mean = np.array(mean)
        std = np.array(std)
        if ret.dtype == np.uint8:
            ret = ret / 255.0
        ret = (ret - mean[None, None, :])/std[None, None, :]
    else:
        raise TypeError('input is not a image.')

    return ret


def denormalize(input, mean, std, inplace=False):
    """Denormalize a image with mean and standard deviation.

    Args:
        input (tensor or numpy.ndarray): image to be denormalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        img: Denormalized image.
    """
    if _is_tensor_image(input):
        if not inplace:
            ret = input.clone()

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        ret.mul_(std[:, None, None]).add_(mean[:, None, None])
    elif _is_numpy_image(input):
        if not inplace:
            ret = input.copy()
        ret = ret * std + mean
        ret = (ret * 255.0).astype(np.uint8)
    else:
        raise TypeError('input is not a image.')

    return ret


def cutout(img,  size):
    """cutout a image with a rectangle(size).

    Args:
        img (tensor or numpy.ndarray): image to be normalized.
        size (int): rectangle size.

    Returns:
        img: cutout image.
    """
    if _is_tensor_image(img):
        _, h,w = img.size()
        idx = random.randint(0, (h-size)*(w-size)-1)
        y = idx // (w-size)
        x = idx - y * (w - size)
        if isinstance(img, torch.ByteTensor):
            img[:, y:y+size, x:x+size] = 128
        else:
            img[:, y:y+size, x:x+size] = 0
    elif _is_numpy_image(img):
        h, w = img.shape[:2]
        idx = random.randint(0, (h-size)*(w-size)-1)
        y = idx // (w-size)
        x = idx - y * (w - size)
        if img.dtype==np.uint8:
            img[y:y+size, x:x+size, :] = 128
        else:
            img[y:y+size, x:x+size, :] = 0
    return img


def resize(img, size, mode='cubic'):
    """Resize the input image to the given size.

    Args:
        img (numpy.ndarray): Image to be resized.
        size (sequence or int): Desired output size.
        mode (string, optional): Desired interpolation mode. Default is 'cubic'

    Returns:
        numpy.ndarray: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    assert interpolation in {'nearest', 'linear', 'cubic'}, 'interpolation mode must be one of nearest, linear, cubic'

    _interp_str_cv2_code_dict = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC
    }

    if isinstance(size, int):
        oh, ow = size, size
    elif isinstance(size, Iterable) and len(size) == 2:
        oh, ow = size[::-1]
    else:
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    return cv2.resize(img, (ow, oh), interpolation=_interp_str_cv2_code_dict[mode])


def pad(img, padding, fill=0, padding_mode='constant'):
    """Pad the given numpy image on all sides with specified padding mode and fill value.

    Args:
        img (numpy.ndarray): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of length 3,
            it is used to fill R, G, B channels respectively.This value is only used when
            the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

    Returns:
        numpy.ndarray: Padded image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric'], \
        'Padding mode should be either constant, edge, reflect or symmetric'

    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, Sequence) and len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    elif isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    else:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple")

    if padding_mode == 'constant':
        if isinstance(fill, numbers.Number):
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_value=fill)
        if isinstance(fill, tuple) and len(fill) == 3:
            raise Exception("not surpport for now!")
        else:
            raise ValueError("fill must be an int or a 3 element tuple")
    else:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

    return img


def crop(img, roi, pad=False):
    """Crop the given numpy image.

    Args:
        img (numpy.ndarray): Image to be cropped.
        roi (list): roi rect
    Returns:
        numpy.array: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = roi
    if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
        if pad:
            top_pad = abs(min(0, ymin))
            bottom_pad = max(ymax - h, 0)
            left_pad = abs(min(0, xmin))
            right_pad = max(xmax - w, 0)
            img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)), mode="constant", constant_values=128)
            ymin += top_pad
            ymax += top_pad
            xmin += left_pad
            xmax += left_pad
        else:
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
    return img[ymin: ymax, xmin: xmax, :]


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.shape[:2]
    th, tw = output_size
    y = int(round((h - th) / 2.))
    x = int(round((w - tw) / 2.))
    return crop(img, [x, y, y+th, x+tw])


def hflip(img):
    """Horizontally flip the given numpy image.

    Args:
        img (numpy.ndarray): Image to be flipped.

    Returns:
        numpy.ndarray:  Horizontall flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    return cv2.flip(img, 1)


def vflip(img):
    """Vertically flip the given numpy image.

    Args:
        img (numpy.ndarray): Image to be flipped.

    Returns:
        numpy.ndarray:  Vertically flipped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return cv2.flip(img, 0)


def blur(img, ksize=3):
    """Blur the given numpy image.

    Args:
        img (numpy.ndarray): Image to be blured.

    Returns:
        numpy.ndarray:  Blured image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    return cv2.blur(img, ksize)

def motion_blur(img, size=15, direction=0):
    # generating the kernel
    if direction==0:
        kernel= np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
    elif direction==1:
        kernel= np.zeros((size, size))
        kernel[:, int((size-1)/2)] = np.ones(size)
    elif direction==2:
        kernel=np.identity(size).astype(float)
    elif direction==3:
        kernel=np.identity(size).astype(float)
        kernel=np.fliplr(kernel)
    else:
        print( "Invalid direction.")
        #sys.exit(-1)
        return img
    kernel= kernel/ size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel)
    return output

def random_lighting(img, light_range=(80, 100), zone=(100, 200), scale=1.5):
    r'''
    This lighting module is based on https://github.com/hujiulong/blog/issues/2
    which
    :param img: [h, w, c] shape
    :param prob: the probability of lighting augmentation
    :param light_range: the lighting color range for each rgb channel
    :param zone: the lighting projected area
    :param scale: the range of lighting center area.
    :return:
    '''

    h, w, c = img.shape
    # TODO: light color for projection, you can change it
    light = [random.randint(light_range[0], light_range[1]) for _ in range(3)]

    # get random light position for lighting
    lcx = random.randint(0, int(w * scale))
    lcy = random.randint(0, int(h * scale))
    # random light zone of lighting, range is bigger when it increase
    rand_zone = random.randint(zone[0], zone[1])
    lightPos = np.array([lcy, lcx, rand_zone]).reshape(1, -1)

    # get image all coord
    yv, xv = np.meshgrid(range(0, h), range(0, w), indexing='ij')
    z = np.zeros((h, w), dtype=np.int32)
    positions = np.stack([yv, xv, z], axis=2)

    # normal vector
    normal = [0, 0, 1]
    # change to int for add op
    img = img.astype(np.int32)
    # add lighting for each pixel
    for hidx in range(h):
        currentLight = sklearnNorm(lightPos - positions[hidx])
        # lighting projection vector
        intensity = np.dot(currentLight, normal).reshape(-1, 1)
        # do lighting change
        img[hidx] = (img[hidx] + light) * intensity

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def adjust_brightness_PIL(img, brightness_factor=[0.7, 1.3]):
    """
    Args:
        img (numpy.ndarray): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2 
    Returns:
        numpy.ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    scale = random.uniform(brightness_factor[0], brightness_factor[1])
    image = ImageEnhance.Brightness(image).enhance(scale)
    img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    
    return img


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (numpy.ndarray): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        numpy.ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    img = img * brightness_factor
    img = np.clip(img, 0, 255)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (numpy.ndarray):Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        numpy.ndarray: Contrast adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = int(gray.mean() + 0.5)
    img = (1.0 - contrast_factor) * mean + contrast_factor * img
    img = np.clip(img, 0, 255)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (numpy.ndarray): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        numpy.ndarray: Saturation adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (1.0 - saturation_factor) * gray + saturation_factor * img
    img = np.clip(img, 0, 255)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (numpy.ndarray): Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        numpy.ndarray: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))

    h, s, v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).split()
    h = h + np.uint8(hue_factor * 180)
    h = (h - 180) if h > 180 else h
    hsv = np.stack([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype(np.uint8)

    # apply gamma correction using the lookup table
    image = image.astype(np.uint8)
    return cv2.LUT(image, table)


def to_grayscale(img, out_channels=1):
    """Convert image to grayscale version of image.

    Args:
        img (numpy.ndarray): Image to be converted to grayscale.

    Returns:
        numpy.ndarray: Grayscale version of the image.
            if out_channels = 1 : returned image is single channel

            if out_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    if out_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif out_channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.stack([img, img, img], axis=0)
    else:
        raise ValueError('out_channels should be either 1 or 3')

    return img


def rotate(img, angle, fillcolor=(128, 128, 128)):
    """Rotate the image by angle.

    Args:
        img (numpy.ndarray):Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
    """

    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    h, w = img.shape[:2]
    cx, cy = w/2+0.5, h/2+0.5
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    img = cv2.warpAffine(img, m, (w, h), borderValue=fillcolor)
    return img


def affine(img, matrix, fillcolor=(0, 0, 0)):
    """Apply affine transformation on the image keeping image center invariant

    Args:
        img (numpy.ndarray):Image to be warped.
        angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float): shear angle value in degrees between -180 to 180, clockwise direction.
        fillcolor (tuple): Optional fill color for the area outside the transform in the output image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))

    h, w = img.shape[:2]
    # print(fillcolor)
    img = cv2.warpAffine(img, matrix, (w, h), borderValue=fillcolor)
    return img

"""======================transform functions for points========================================"""

# normalize pts from world coord to range[-1, 1]
def normalize_pts(pts, h, w):
    pts = 2*pts/np.array([w-1, h-1]) - 1.0
    return pts


# denormalize pts to world coord
def denormalize_pts(pts, h, w):
    pts = (pts+1.0)*np.array([w-1, h-1])/2.0
    return pts


def inverse_transform_pts(pts, theta):
    if isinstance(pts, np.ndarray) and isinstance(theta, np.ndarray):
        s = np.ones((68, 3), dtype=np.float)
        s[:, :2] = pts[:, :]
        pts = np.matmul(s, matrix.T)

    elif isinstance(pts, torch.Tensor) and isinstance(theta, torch.Tensor):
        pts = torch.nn.functional.pad(pts, (0, 1), mode='constant', value=1) # [n, pt_num, 3]
        pts = torch.bmm(pts, theta.transpose(1, 2)) # [n, pt_num, 3]*[n, 3, 2] = [n, pt_num, 2]

    else:
        raise Exception('type error!')
    return pts

def check_pt_in_img(pts, h, w):
    idx = pts[:, 0] < -1
    pts[idx, 0] = -1
    idx = pts[:, 1] < -1
    pts[idx, 1] = -1
    idx = pts[:, 0] > w+1
    pts[idx, 0] = w+1
    idx = pts[:, 1] > h+1
    pts[idx, 1] = h+1
    return pts

"""======================transform functions for image and pts pair========================="""
def resize_pair(img, pt, size):
    if isinstance(size, numbers.Number):
        size = (size, size)

    # resize to the target size
    th, tw  = size
    sh, sw = img.shape[:2]
    img = cv2.resize(img, (tw, th))
    pt = (pt + 0.5) / np.array([sw, sh]) * np.array([tw, th]) - 0.5
    return img, pt

def crop_pair(img, pt, roi, pad=False):
    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = roi
    top_pad, bottom_pad, left_pad, right_pad = 0, 0, 0, 0
    if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
        if pad:
            top_pad = abs(min(0, ymin))
            bottom_pad = max(ymax - h, 0)
            left_pad = abs(min(0, xmin))
            right_pad = max(xmax - w, 0)
            img = np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0,0)), mode="constant", constant_values=128)
            ymin += top_pad
            ymax += top_pad
            xmin += left_pad
            xmax += left_pad
        else:
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

    img = img[ymin: ymax, xmin: xmax, :]
    pt = pt + np.array([left_pad, top_pad]) - np.array([xmin, ymin])
    return img, pt

def affine_pair(img, pts, matrix, fillcolor=(128, 128, 128)):
    h, w = img.shape[:2]
    img = cv2.warpAffine(img, matrix, (w, h), borderValue=fillcolor)
    n = len(pts)
    s = np.ones((n, 3), dtype=np.float)
    s[:, :2] = pts[:, :]
    pts = np.matmul(s, matrix.T)
    return img, pts

def rotate_pair(img, pts, angle, fillcolor=(0, 0, 0)):
    """Rotate the image by angle.

    Args:
        img (numpy.ndarray):Image to be rotated.
        angle (float or int): In degrees degrees counter clockwise order.
    """

    if not _is_numpy_image(img):
        raise TypeError('img should be numpy image. Got {}'.format(type(img)))
    h, w = img.shape[:2]
    cx, cy = w/2+0.5, h/2+0.5
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    img = cv2.warpAffine(img, m, (w, h), borderValue=fillcolor)

    n = len(pts)
    s = np.ones((n, 3), dtype=np.float)
    s[:, :2] = pts[:, :]
    pts = np.matmul(s, m.T)
    return img, pts

"""=======================transform functions for bounding box=============================="""
def extend(bbox, ratio, img_size):
    if isinstance(ratio, Number):
        ratio=(ratio, ratio)

    h, w = img_size
    xmin, ymin, xmax, ymax = bbox
    bh, bw = ymax - ymin, xmax - xmin
    xmin = int(max(0, xmin-bw*ratio[0]))
    ymin = int(max(0, ymin-bh*ratio[1]))
    xmax = int(min(w-1, xmax+bw*ratio[0]))
    ymax = int(min(h-1, ymax+bh*ratio[1]))
    return [xmin, ymin, xmax, ymax]


def bounding_box(pts):
    xmin = np.min(pts[:, 0])
    ymin = np.min(pts[:, 1])
    xmax = np.max(pts[:, 0])
    ymax = np.max(pts[:, 1])
    return [xmin, ymin, xmax, ymax]


def match_bbox(bboxes, ref):
    max_overlap = IOU(bboxes[0], ref)
    matched = bboxes[0]
    for bbox in bboxes[1:]:
        overlap = IOU(bbox, ref)
        if overlap > max_overlap:
            max_overlap = overlap
            matched = bbox
    return max_overlap, matched


def IOU(bbox0, bbox1):
    xmin = min(bbox0[0], bbox1[0])
    ymin = min(bbox0[1], bbox1[1])
    xmax = max(bbox0[2], bbox1[2])
    ymax = max(bbox0[3], bbox1[3])
    w0, h0 = bbox0[2] - bbox0[0], bbox0[3] -bbox0[1]
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] -bbox1[1]
    w, h = w0+w1-(xmax-xmin), h0+h1-(ymax-ymin)
    if w <= 0 or h <= 0:
        return 0
    else:
        return 1.0*w*h/(w0*h0+w1*h1-w*h)

"""======================affine matrix construct function==================================="""

def inverse_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute inverse matrix for affine transformation

    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    angle = math.radians(angle)
    shear = math.radians(shear)
    scale = 1.0 / scale

    # Inverted rotation matrix with scale and shear
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
    matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    return np.array(matrix).reshape(2, 3)
