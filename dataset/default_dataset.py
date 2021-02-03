"""WiderFace dataset for image, bboxes, points, labels
@author: mingchao jiang
@date  : 2021-01-26
"""
import os
import cv2
import json
import torch
import random
import numpy as np
import urllib.request as urt

from PIL import Image
from io import BytesIO
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

# from transforms.transforms import Resize, RandomDistort, RandomFlip, Compose

class BaseDataSet(Dataset):
    def __init__(self, cfg, txt_file, augment_transform=None, batch_transfrom=None, read_format="cv2", min_area=1, use_ldm=False):
        """Widerface base dataset
        Args:
            txt_file : a file for json lines, each json lines is like this
                        {'image_path': xxx.jpg,  'image_bbox': [x1, y1, x2, y2]...,  'image_pts': [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5] or [-1] * 10 ... }
            base_transform : resize with normalize for image, bbox, pts, use for train or eval
            augment_transform: augment for resize, flip, distort, random crop..etc
            batch_transform: used only for mosaic or other need mutil image augment method
            read_format: "cv2" or "PIL"
            min_area: the min bbox area to ignore
            use_ldm: Train with the landmarks or not
        """
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.txt_file = txt_file
        self.use_ldm = use_ldm
        if self.use_ldm:
            self.target_dim = 4 + 1 + 10
        else:
            self.target_dim = 4 + 1
        txt_list = [json.loads(x.strip()) for x in open(self.txt_file).readlines()]
        self.samples = []
        for i in range(len(txt_list)):
            image_path = txt_list[i]['image_path']
            image_bbox = txt_list[i]['image_bbox']
            image_pts = txt_list[i]['image_pts']
            if self.use_ldm:
                self.samples.append([image_path, image_bbox, image_pts])
            else:
                self.samples.append([image_path, image_bbox, None])

        self.mean = torch.Tensor(self.cfg.MODEL.MEAN).float().view(3, 1, 1)
        self.std = torch.Tensor(self.cfg.MODEL.STD).float().view(3, 1, 1)

        self.augment_transform = None
        self.batch_transfrom = batch_transfrom
        self.min_area = min_area

        self.read_format = read_format
        random.seed(13)

    def base_transform(self, x):
        return (x - self.mean) / self.std

    def _load_image(self, image_file, read_format='cv2', num_trys=20, color_type="RGB"):
        """
        Args:
            image_file   : xxx.jpg
            read_format  : "PIL" or "cv2"
            num_trys     : try numbers for read file
            color_type   : "RGB" or "BGR"
        Returns:
            image : a Numpy array , the order is BGR
        """
        if "http" in image_file:
            for _ in range(num_trys):
                try:
                    image_content = urt.urlopen(image_file).read()
                    if read_format.upper() == "PIL":
                        image = Image.open(BytesIO(image_content))
                        if color_type is "RGB":
                            image = image.convert("RGB")
                        else:
                            image = image.convert("BGR")
                        image = np.array(image)
                        return image
                    elif read_format == "cv2":
                        image = cv2.imdecode(np.asarray(bytearray(image_content), dtype="uint8"), cv2.IMREAD_COLOR)
                        if color_type is "RGB":
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        return image
                    else:
                        raise RuntimeError("Read format must be pil or cv2")
                except Exception as e:
                    print(f"Make a exception {e}, data file is {image_file}")
        else:
            for _ in range(num_trys):
                try:
                    if read_format.upper() == "PIL":
                        image = Image.open(image_file)
                        if color_type is "RGB":
                            image = image.convert("RGB")
                        else:
                            image = image.convert("BGR")
                        image = np.array(image)
                        return image
                    elif read_format == "cv2":
                        if color_type is "RGB":
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        return image
                    else:
                        raise RuntimeError("Read format must be pil or cv2")
                except Exception as e:
                    print(f"Make a exception {e}, data file is {image_file}")

    def _make_target(self, bbox, points):
        """return array target
           no points (N, 5) or with points (N, 15) ndarray
        """
        labels = 0
        target_output = []
        for i in range(len(bbox)):
            x1, y1, x2, y2 = bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]
            if points is None:
                target_output.append([x1, y1, x2, y2, labels])
            else:
                target_output.append([x1, y1, x2, y2, labels] + points[i])
        # N X (4 + 1 + 10)
        target_array = np.array(target_output)
        return target_array

    def __getitem__(self, idx):
        """transform image, bbox, labels, points to tensor, augment and ignore the min area.
        Returns:
            image  : torch.tensor
            bboxes : torch.tensor
            labels : torch.LongTensor
            points : torch.tensor
            has_pts: torch.booltensor
        """
        img_path, img_bbox, img_pts = self.samples[idx]
        image = self._load_image(img_path)
        image = cv2.resize(image, (640, 640))
        # only used for test
        # N x (4 + 1) or N x (4 + 1 + 10)
        target = self._make_target(img_bbox, img_pts).reshape(-1, self.target_dim)

        assert image is not None and target is not None, f"Sample {img_path} have no problem!!!"
        h, w = image.shape[0], image.shape[1]

        # x1, y1, x2, y2
        bboxes = target[:, :4]
        labels = target[:, 4]
        pts = None
        has_pts = None
        # x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 for face points
        if img_pts is not None:
            pts = target[:, 5:15]
            has_pts = pts.sum(axis=1) > 0
        # target tensor
        bboxes, labels, pts, has_pts = self._convert_tensor(bboxes, labels, pts, has_pts)

        if type(image) == np.ndarray:
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float)
        # totensor with normalize
        if self.base_transform is not None:
            image = self.base_transform(image)

        # data augment transform
        # TODO: augment transform with points or no points
        if self.augment_transform is not None:
            image, bboxes, labels, pts, has_pts = self.augment_transform(image, bboxes, labels, pts, has_pts)

        # batch augment transform
        if self.batch_transfrom is not None:
            image, bboxes, labels, pts, has_pts = self.augment_transform(self.samples, image, bboxes, labels, pts, has_pts)

        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # ignore bbox area small than min_area, default min_area = 1
        area_mask = area > self.min_area * self.min_area
        area_mask = area_mask.float()
        labels[area < self.min_area * self.min_area] = -1
        bboxes[area < self.min_area * self.min_area] = 0.0
        if pts is not None:
            pts[area < self.min_area * self.min_area] = -1
            has_pts[area < self.min_area * self.min_area] = -1

        return image, labels, bboxes, pts, has_pts, area_mask

    def _convert_tensor(self, bboxes, labels, pts, has_pts):
        """convert targets to tensor"""
        bboxes_tensor = torch.Tensor(bboxes).float()
        labels_tensor = torch.LongTensor(labels)
        if pts is not None and has_pts is not None:
            pts_tensor = torch.Tensor(pts).float()
            has_pts_tensor = torch.BoolTensor(has_pts)
            return bboxes_tensor, labels_tensor, pts_tensor, has_pts_tensor
        else:
            return bboxes_tensor, labels_tensor, None, None

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    data_file = "/data/remote/MarkCamera/facedataset/training/face_annoation_1210_for_train.txt"
    # base_transform = transforms.ToTensor()
    class ToTensor(object):
        def __init__(self):
            self.to_tensor = transforms.ToTensor()
        def __call__(self, img, boxes, labels, pts, has_pt):
            img = cv2.resize(img, (640, 640))
            img = self.to_tensor(img)
            return img, boxes, labels, pts, has_pt

    base_transform = ToTensor()
    dataset = BaseDataSet(data_file, base_transform=None)

    # for idx, data in enumerate(dataset):
    #     print(idx, data["classes"].shape)
    from collate_function import collate_fn
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fn

    )
    from utils.build_targets import Targets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = Targets(device)
    for idx, data in enumerate(loader):
        image, boxes, labels, pts, pts_mask, instance_mask = data
        gt = targets.prepare_targets(image, boxes, labels, pts, pts_mask, instance_mask)
        print(gt)
