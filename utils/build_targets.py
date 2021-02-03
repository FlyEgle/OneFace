"""
build the targets
"""

import torch
import torch.nn.functional as F

from torchvision.transforms import transforms
from utils.box_ops import box_xyxy_to_cxcywh


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
    def __call__(self, img, boxes, labels, pts, has_pt, instance_mask):
        # img = self.to_tensor(img)
        img = self.normalize(img)
        return img, boxes, labels, pts, has_pt, instance_mask


class Targets(object):
    def __init__(self, device):
        super(Targets, self).__init__()
        self.device = device
        self.to_tensor = ToTensor()

    def preprocess_image(self, batch_inputs):
        images, bboxes, labels, pts, has_pt, instance_mask = self.to_tensor(*batch_inputs)
        return images, bboxes, labels, pts, has_pt, instance_mask

    def prepare_targets(self, batch_inputs):
        """ prepare the training targets, this version is keep the w and h is the same
        Args:
            batch_inputs:
                images  : a batch of images tensor -> (B, 3, H, W)
                bboxes  : a batch of bboxes tensor -> (B, N, 4)
                labels  : a batch of labels tensor -> (B, num_classes)
                points  : a batch of points tensor -> (B, N, 10) or None
                has_pts : a batch of has_pts tensor -> (B, N, 1) or None
                instance_mask : a batch of instance_mask tensor -> (B, N, 1)
        Returns:
            targets : a dict for images, bboxes, bboxes_cxcywh, pts, has_pts, instance_mask, labels

        """
        images, bboxes, labels, points, has_pts, instance_mask = batch_inputs
        h, w = images[0].shape[1:]
        image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
        target_images = images.to(self.device)
        target_classes = labels.to(self.device)
        # normalize bboxes with wdith and height and convert the x1, y1, x2, y2 to centerx, centery, w, h
        target_boxes = bboxes / image_size_xyxy
        target_boxes_cxcywh = box_xyxy_to_cxcywh(target_boxes).to(self.device)
        # x1, y1, x2, y2 bboxes
        bboxes = bboxes.to(self.device)
        # images w, h, w, h tensor shape -> (batch, Num)
        target_image_size_xyxy = torch.stack([torch.cat([image_size_xyxy]) for _ in range(images.shape[0])], dim=0)
        target_image_size_xyxy = target_image_size_xyxy.to(self.device)
        # instance mask
        target_instance_mask = instance_mask.to(self.device)
        if points[0] is not None:
            target_points = points.to(self.device)
        else:
            target_points = points

        if has_pts[0] is not None:
            target_has_pts = has_pts.to(self.device)
        else:
            target_has_pts = has_pts

        targets = {}
        targets['images'] = target_images
        targets['labels'] = target_classes
        targets['bboxes'] = target_boxes_cxcywh   # (centerx, centery, w, h)
        targets['boxes_xyxy'] =  bboxes     # (x1, y1, x2, y2)
        targets['image_size_xyxy'] = target_image_size_xyxy
        targets['pts'] = target_points
        targets['pts_mask'] = has_pts
        targets['instance_mask'] = target_instance_mask

        return targets
