import torch 
import cv2 

from dataset.default_dataset import BaseDataSet
from torch.utils.data import DataLoader
from dataset.collate_function import collate_fn
from utils.build_targets import Targets
from torchvision.transforms import transforms

from models.detector import OneFace
from loss.loss import SetCriterion, MinCostMatcher

model = OneFace("resnet50", pretrain=True)
model.cuda()
print(model)

data_file = "/data/remote/MarkCamera/facedataset/training/face_annoation_1210_for_train.txt"
# base_transform = transforms.ToTensor()
dataset = BaseDataSet(data_file, base_transform=None)

# for idx, data in enumerate(dataset):
#     print(idx, data["classes"].shape)

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn,
    shuffle=False
)

class_weight = 2
giou_weight = 2
l1_weight =5
matcher = MinCostMatcher(cost_class=class_weight,
                         cost_bbox=l1_weight,
                         cost_giou=giou_weight)
weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
# loss 
losses = ["labels", "boxes"]
criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
targets = Targets(device)
for idx, data in enumerate(loader):
    image, boxes, labels, pts, pts_mask, instance_mask = targets.preprocess_image(data) 
    # print(boxes)
    target = targets.prepare_targets(image, boxes, labels, pts, pts_mask, instance_mask)
    image = image.to(device)
    outputs_class, outputs_coord, outputs_pts = model(image)
    output = {'pred_logits': outputs_class,  'pred_boxes': outputs_coord}
    # indices = matcher(output, target)

    loss = criterion(output, target)
    print(loss)


