'''Transforms for training the face detection
'''
from __future__ import division
import torch
import torchvision.transforms as torch_transforms
import math
import random   #do not use numpy's random libraries for subprocess behavior
import numpy as np
from transforms import functional as F
import collections
from PIL import Image
import cv2
import urllib
from utils.box_utils import box_iou, box_clamp, pts_clamp

__all__ = ["Compose", "ToTensor", "Normalize", "RandomDistort", "RandomGrayscale",
           "RandomPaste", "RandomFlip", "RandomCrop", "Resize", "RandomMotionBlur",
           "RandomGaussianNoise", "RandomAdjustGamma", "RandomCutout", "Mosaic"]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes, labels, pts, has_pt):
        for k, t in enumerate(self.transforms):
            img, boxes, labels, pts, has_pt = t(img, boxes, labels, pts, has_pt)
        return img, boxes, labels, pts, has_pt

class ToTensor(object):
    def __init__(self):
        self.to_tensor = torch_transforms.ToTensor()
    def __call__(self, img, boxes, labels, pts, has_pt):
        img = self.to_tensor(img)
        return img, boxes, labels, pts, has_pt

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.norm = torch_transforms.Normalize(self.mean, self.std)
    def __call__(self, img, boxes, labels, pts, has_pt):
        img = self.norm(img)
        return img, boxes, labels, pts, has_pt

class RandomDistort(object):
    def __init__(self, brightness_delta=32/255., contrast_delta=0.5, saturation_delta=0.5, hue_delta=0.1):
        '''A color related data augmentation used in SSD.
        Args:
          img: (PIL.Image) image to be color augmented.
          brightness_delta: (float) shift of brightness, range from [1-delta,1+delta].
          contrast_delta: (float) shift of contrast, range from [1-delta,1+delta].
          saturation_delta: (float) shift of saturation, range from [1-delta,1+delta].
          hue_delta: (float) shift of hue, range from [-delta,delta].
        Returns:
          img: (PIL.Image) color augmented image.
        '''
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.saturation_delta = saturation_delta
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels, pts, has_pt):
        def brightness(img, delta):
            if random.random() < 0.5:
                img = torch_transforms.ColorJitter(brightness=delta)(img)
            return img

        def contrast(img, delta):
            if random.random() < 0.5:
                img = torch_transforms.ColorJitter(contrast=delta)(img)
            return img

        def saturation(img, delta):
            if random.random() < 0.5:
                img = torch_transforms.ColorJitter(saturation=delta)(img)
            return img

        def hue(img, delta):
            if random.random() < 0.5:
                img = torch_transforms.ColorJitter(hue=delta)(img)
            return img

        img = brightness(img, self.brightness_delta)
        if random.random() < 0.5:
            img = contrast(img, self.contrast_delta)
            img = saturation(img, self.saturation_delta)
            img = hue(img, self.hue_delta)
        else:
            img = saturation(img, self.saturation_delta)
            img = hue(img, self.hue_delta)
            img = contrast(img, self.contrast_delta)
        return img, boxes, labels, pts, has_pt

class RandomGrayscale(object):
    def __init__(self, proba=0.3):
        self._proba = proba

    def __call__(self, img, boxes, labels, pts, has_pt):
        if random.random() < self._proba:
            img = img.convert("L")
            img = np.array(img)[:, :, None]
            img = np.tile(img, (1, 1, 3))
            img = Image.fromarray(img)

        return img, boxes, labels, pts, has_pt

class RandomPaste(object):
    def __init__(self, max_ratio=4, fill=0, proba=1.):
        '''Randomly paste the input image on a larger canvas.
        If boxes is not None, adjust boxes accordingly.
        Args:
          img: (PIL.Image) image to be flipped.
          boxes: (tensor) object boxes, sized [#obj,4].
          pts: (tensor) object 5_pts, sized [#obj,10].
          max_ratio: (int) maximum ratio of expansion.
          fill: (tuple) the RGB value to fill the canvas.
        Returns:
          canvas: (PIL.Image) canvas with image pasted.
          boxes: (tensor) adjusted object boxes.
          pts: (tensor) adjusted object pts.
        '''
        self.max_ratio = max_ratio
        self.fill = fill
        self.proba = proba

    def __call__(self, img, boxes, labels, pts, has_pt):
        if random.uniform(0, 1) < self.proba:
            w, h = img.size
            ratio = random.uniform(1, self.max_ratio)
            ow, oh = int(w*ratio), int(h*ratio)
            canvas = Image.new('RGB', (ow, oh), self.fill)

            x = random.randint(0, ow - w)
            y = random.randint(0, oh - h)
            canvas.paste(img, (x, y))

            if boxes is not None:
                boxes = boxes + torch.tensor([x, y]*2, dtype=torch.float)
            if pts is not None:
                pts = pts + torch.tensor([x, y]*5, dtype=torch.float)
            return canvas, boxes, labels, pts, has_pt
        else:
            return img, boxes, labels, pts, has_pt

class RandomFlip(object):
    def __init__(self, proba=0.5):
        '''Randomly flip PIL image.
        If boxes is not None, flip boxes accordingly.
        Args:
          img: (PIL.Image) image to be flipped.
          boxes: (tensor) object boxes, sized [#obj,4].
          pts: (tensor) object pts, sized [#obj,10].
        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped boxes.
          pts: (tensor) randomly flipped pts.
        '''
        self.proba = proba

    def __call__(self, img, boxes, labels, pts, has_pt):
        if random.random() < self.proba:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            if boxes is not None:
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax

            if pts is not None:
                last_pts = pts.clone()
                l_eye_x = w - last_pts[:, 2]
                l_eye_y = last_pts[:, 3]
                r_eye_x = w - last_pts[:, 0]
                r_eye_y = last_pts[:, 1]
                nose    = w - last_pts[:, 4]
                l_mouse_x = w - last_pts[:, 8]
                l_mouse_y = last_pts[:, 9]
                r_mouse_x = w - last_pts[:, 6]
                r_mouse_y = last_pts[:, 7]
                pts[:, 0] = l_eye_x
                pts[:, 1] = l_eye_y
                pts[:, 2] = r_eye_x
                pts[:, 3] = r_eye_y
                pts[:, 4] = nose
                pts[:, 6] = l_mouse_x
                pts[:, 7] = l_mouse_y
                pts[:, 8] = r_mouse_x
                pts[:, 9] = r_mouse_y

        return img, boxes, labels, pts, has_pt

class RandomCrop(object):
    def __init__(self, min_scale=0.3, max_aspect_ratio=2., proba=1.):
        self.min_scale = min_scale
        self.max_aspect_ratio = max_aspect_ratio
        self.proba = proba

    def __call__(self, img, boxes, labels, pts, has_pt):
        if random.random() > self.proba:
            return img, boxes, labels, pts, has_pt

        imw, imh = img.size
        params = [(0, 0, imw, imh)]  # crop roi (x,y,w,h) out
        min_iou=random.choice([0, 0.1, 0.3, 0.5, 0.7, 0.9])
        #for min_iou in (0, 0.1, 0.3, 0.5, 0.7, 0.9):
        for _ in range(10):
            scale = random.uniform(self.min_scale, 1)
            aspect_ratio = random.uniform(
                max(1/self.max_aspect_ratio, scale*scale),
                min(self.max_aspect_ratio, 1/(scale*scale)))
            w = int(imw * scale * math.sqrt(aspect_ratio))
            h = int(imh * scale / math.sqrt(aspect_ratio))

            x = random.randrange(imw - w)
            y = random.randrange(imh - h)

            roi = torch.tensor([[x, y, x+w, y+h]], dtype=torch.float)
            ious = box_iou(boxes, roi)
            params.append((x, y, w, h))

            if ious.min() >= min_iou:
                params=[(x, y, w, h)]
                break

        x, y, w, h = random.choice(params)
        img = img.crop((x, y, x+w, y+h))

        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] >= x) & (center[:, 0] <= x+w) \
            & (center[:, 1] >= y) & (center[:, 1] <= y+h)
        if mask.any():
            boxes = boxes[mask] - torch.tensor([x, y]*2, dtype=torch.float)
            boxes = box_clamp(boxes, 0, 0, w, h)
            labels = labels[mask]
            pts = pts[mask] - torch.tensor([x, y]*5, dtype=torch.float)
            # pts = pts_clamp(pts, 0, 0, w, h)
            has_pt = has_pt[mask]
        else:
            boxes = torch.tensor([[0, 0]*2], dtype=torch.float)
            labels = torch.tensor([-1], dtype=torch.long)
            pts = torch.tensor([[0, 0]*5], dtype=torch.float)
            has_pt = torch.tensor([False], dtype=torch.bool)
        return img, boxes, labels, pts, has_pt


class Resize(object):
    def __init__(self, size, max_size=1000, random_interpolation=False):
        '''Resize the input PIL image to given size.
        If boxes is not None, resize boxes accordingly.
        Args:
          img: (PIL.Image) image to be resized.
          boxes: (tensor) object boxes, sized [#obj,4].
          pts: (tensor) object pts, sized [#obj,10].
          size: (tuple or int)
            - if is tuple, resize image to the size.
            - if is int, resize the shorter side to the size while maintaining the aspect ratio.
          max_size: (int) when size is int, limit the image longer size to max_size.
                    This is essential to limit the usage of GPU memory.
        Returns:
          img: (PIL.Image) resized image.
          boxes: (tensor) resized boxes.
        Example:
        >> img, boxes, pts = resize(img, boxes, 600)  # resize shorter side to 600
        >> img, boxes, pts = resize(img, boxes, (500,600))  # resize image size to (500,600)
        >> img, _, _ = resize(img, None, None, (500,600))  # resize image only
        '''
        self.size=size
        self.max_size = max_size
        self.random_interpolation = random_interpolation

    def __call__(self, img, boxes, labels, pts, has_pt):
        w, h = img.size
        if isinstance(self.size, int):
            size_min = min(w, h)
            size_max = max(w, h)
            sw = sh = float(self.size) / size_min
            if sw * size_max > self.max_size:
                sw = sh = float(self.max_size) / size_max
            ow = int(w * sw + 0.5)
            oh = int(h * sh + 0.5)
        else:
            ow, oh = self.size
            sw = float(ow) / w
            sh = float(oh) / h

        method = random.choice([
        Image.BOX,
        Image.NEAREST,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        Image.BILINEAR]) if self.random_interpolation else Image.BILINEAR
        img = img.resize((ow, oh), method)
        if boxes is not None:
            boxes = boxes * torch.tensor([sw, sh]*2)
        if pts is not None:
            pts = pts * torch.tensor([sw, sh]*5)

        return img, boxes, labels, pts, has_pt

class RandomMotionBlur(object):
    def __init__(self, blur_intensity, direction, proba=0.3):
        self.blur_intensity=blur_intensity
        self.direction=direction
        self.proba=proba

    def __call__(self, image, boxes, labels, pts, has_pt):
        image=np.array(image)
        if random.uniform(0,1)<self.proba:
            if isinstance(self.blur_intensity, collections.Sequence):
                blur_intensity=random.randint(self.blur_intensity[0], self.blur_intensity[1])
            else:
                blur_intensity= self.blur_intensity

            if isinstance(self.direction, collections.Sequence):
                direction=random.choice(self.direction)
            else:
                direction=direction

            image = F.motion_blur(image, blur_intensity, direction)

        image=Image.fromarray(image)
        return image, boxes, labels, pts, has_pt

class RandomGaussianNoise(object):
    """Add gaussian noise to a numpy.ndarray (H x W x C)
    """
    def __init__(self, mean, sigma, proba=0.3):
        self.sigma = sigma
        self.mean = mean
        self.proba = proba

    def __call__(self, image, boxes, labels, pts, has_pt):
        if random.uniform(0, 1) < self.proba:
            image = np.array(image)
            mean, sigma = self.__gen_param()

            row, col, ch = image.shape
            gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(float)
            gauss = gauss.reshape(row, col, ch)
            image = image.astype(float)
            image += gauss

            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)

            image = Image.fromarray(image)

        return image, boxes, labels, pts, has_pt

    def __gen_param(self):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
        else:
            sigma = self.sigma

        if isinstance(self.mean, collections.Sequence):
            mean = random.uniform(self.mean[0], self.mean[1])
        else:
            mean = self.mean

        return mean, sigma

class RandomAdjustGamma(object):
    def __init__(self, gamma=0.6, proba=0.3):
        self.gamma = gamma
        self.proba=proba

    def __call__(self, img, boxes, labels, pts, has_pt):
        """img: pil image"""
        if random.uniform(0,1)<self.proba:
            img = np.array(img)
            g = self.__gen_param()
            img = F.adjust_gamma(img, g)
            img = Image.fromarray(img)

        return img, boxes, labels, pts, has_pt

    def __gen_param(self):
        if isinstance(self.gamma, collections.Sequence):
            g = random.uniform(self.gamma[0], self.gamma[1])
        else:
            g = random.uniform(1-self.gamma, 1+self.gamma)
        return g

class RandomCutout(object):
    def __init__(self, min_radio=0.4, max_radio=0.7, p=0.3):
        self.min_radio = min_radio
        self.max_radio = max_radio
        self.p = p

    def __call__(self, img, boxes, labels, pts, has_pt):
        img = np.array(img)
        (h, w) = img.shape[:2]
        for box in boxes:
            if random.random() > self.p:
                continue

            b_w, b_h = box[2]-box[0], box[3]-box[1]
            box_scale = min(b_w//w, b_h//h)
            if box_scale < 0.1:
                continue

            radio_w = random.uniform(self.min_radio, self.max_radio)
            radio_h = random.uniform(self.min_radio, self.max_radio)
            cut_w = b_w * radio_w
            cut_h = b_h * radio_h
            idx = random.randint(0, int((b_w-cut_w)*(b_h-cut_h)-1))
            y = int(box[1] + idx // (b_w-cut_w))
            x = int(box[0] + idx - (y-box[1]) * (b_w-cut_w))
            img[y:int(y+cut_h), x:int(x+cut_w), :] = 128

        img = Image.fromarray(img)
        return img, boxes, labels, pts, has_pt


class Mosaic(object):
    def __init__(self, min_offset=0.35, proba=0.35, transforms=None):
        self.min_offset = min_offset
        self.proba = proba
        self.transforms = transforms

    def __call__(self, sample_list, img, boxes, labels, pts, has_pt):
        if random.random() > self.proba:
            return img, boxes, labels, pts, has_pt

        img = np.array(img)
        inds = [random.choice(list(range(len(sample_list)))) for i in range(3)]
        new_imgs, new_boxes, new_labels, new_pts, new_has_pts = [img], [boxes], [labels], [pts], [has_pt]
        for ind in inds:
            new_img, new_box, new_label, new_pt, new_has_pt = self.transforms_one_img(sample_list, ind)
            new_imgs.append(np.array(new_img).copy())
            new_boxes.append(new_box.clone())
            new_labels.append(new_label.clone())
            new_pts.append(new_pt.clone())
            new_has_pts.append(new_has_pt.clone())

        image_h, image_w = img.shape[:2]
        cut_x = np.random.randint(int(image_w*self.min_offset), int(image_w*(1 - self.min_offset)))
        cut_y = np.random.randint(int(image_h*self.min_offset), int(image_h*(1 - self.min_offset)))

        img_szs = []
        img_szs.append((cut_x, cut_y))
        img_szs.append((image_w-cut_x, cut_y))
        img_szs.append((cut_x, image_h-cut_y))
        img_szs.append((image_w-cut_x, image_h-cut_y))
        sws = []
        sws.append([cut_x/image_w, cut_y/image_h])
        sws.append([1-cut_x/image_w, cut_y/image_h])
        sws.append([cut_x/image_w, 1-cut_y/image_h])
        sws.append([1-cut_x/image_w, 1-cut_y/image_h])
        xy_shifts = []
        xy_shifts.append([0, 0])
        xy_shifts.append([cut_x, 0])
        xy_shifts.append([0, cut_y])
        xy_shifts.append([cut_x, cut_y])

        out_img = img.copy()

        for i in range(len(new_imgs)):
            new_imgs[i] = cv2.resize(new_imgs[i], img_szs[i])
            new_boxes[i] = new_boxes[i] * torch.tensor(sws[i]*2) + torch.tensor(xy_shifts[i]*2)
            new_pts[i] = new_pts[i] * torch.tensor(sws[i]*5) + torch.tensor(xy_shifts[i]*5)

        out_img[0:cut_y, 0:cut_x, :] = new_imgs[0]
        out_img[0:cut_y, cut_x:image_w, :] = new_imgs[1]
        out_img[cut_y:image_h, 0:cut_x, :] = new_imgs[2]
        out_img[cut_y:image_h, cut_x:image_w, :] = new_imgs[3]

        for k in range(1, len(new_imgs)):
            new_boxes[0] = torch.cat((new_boxes[0], new_boxes[k]), 0)
            new_labels[0] = torch.cat((new_labels[0], new_labels[k]), 0)
            new_pts[0] = torch.cat((new_pts[0], new_pts[k]), 0)
            new_has_pts[0] = torch.cat((new_has_pts[0], new_has_pts[k]), 0)

        return out_img, new_boxes[0], new_labels[0], new_pts[0], new_has_pts[0]


    def transforms_one_img(self, samples, index):
        img_path, annot_path = samples[index]
        if img_path[0:4] == 'http':
            img = Image.open(urllib.request.urlopen(img_path)).convert('RGB')
            img = np.asarray(img)
            pts = []
            with urllib.request.urlopen(annot_path) as f:
                for line in f.readlines():
                    items = str(line, encoding = 'utf-8').split()
                    items[-1] = items[-1].rstrip()
                    pts.append(np.array([float(p) for p in items]))

            target = np.zeros((len(pts), 15), dtype=np.float)
            for p_i, pt in enumerate(pts):
                target[p_i] = pt

        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            target = np.loadtxt(annot_path, dtype=np.float32)
            target = target.reshape(-1, 15)

        assert img is not None and target is not None, 'sample {} have problem!'.format(img_path)
        h, w = img.shape[:2]
        boxes = target[:, :4]
        # x1, y1, w, h -> x1, y1, x2, y2
        boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
        labels = target[:, 4]
        pts = target[:, 5:15]
        has_pt = pts.sum(axis=1) > 0

        # convert to torch tensor
        boxes = torch.Tensor(boxes)
        labels = torch.LongTensor(labels)
        pts = torch.Tensor(pts)
        has_pt = torch.BoolTensor(has_pt)

        img = Image.fromarray(img)
        if self.transforms is not None:
            img, boxes, labels, pts, has_pt = self.transforms(img, boxes, labels, pts, has_pt)

        return img, boxes, labels, pts, has_pt


