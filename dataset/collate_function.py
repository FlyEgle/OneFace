import torch
import numpy as np


def collate_fn(outputs):
    """Create batch from multiple samples, get the max detection bbox smaple, cat 0 for small to get batch
    Returns:
        image: a batch image tensor
        boxes: a batch boxes tensor
        labels: a batch labels tensor
        pts: a batch points tensor or None
        has_pts: a batch points mask or None
        instance_mask : a batch instance mask for object and labels
    """
    image, labels, boxes, pts, has_pts, bbox_mask = zip(*outputs)
    # get the max dets from a image
    max_det = max([t.size()[0] for t in boxes])  # bs x N x 4
    # instance real numbers
    bboxes_mask = []
    for i in range(len(bbox_mask)):
        bboxes_mask.append(torch.cat([bbox_mask[i], torch.ones(max_det - bbox_mask[i].size()[0]) * 0]))
    instance_mask = torch.stack(bboxes_mask, 0)
    boxes = [torch.cat([t, torch.ones([max_det - t.size()[0], 4]) * 0]) for t in boxes]
    boxes = torch.stack(boxes, 0)
    labels = [torch.cat([t, torch.ones([max_det - t.size()[0]]).long() * -1]) for t in labels]
    labels = torch.stack(labels, 0)
    if pts[0] is not None:
        pts = [torch.cat([t, torch.ones([max_det - t.size()[0], 10]) * -1]) for t in pts]
        pts = torch.stack(pts, 0)
    if has_pts[0] is not None:
        has_pts = [torch.cat([t, (torch.ones([max_det - t.size()[0]]) * 0).bool()]) for t in has_pts]
    image = torch.stack(image, 0)
    return image, boxes, labels, pts, has_pts, instance_mask


