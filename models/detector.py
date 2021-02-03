"""
Build the detection detector
make with the backbone, fpn, head
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# backbone
from models.resnet import resnet50, resnet101
# from models.resnetd import resnet50d, resnet101d


def build_backbone(backbone_name):
    """get backbone"""
    if backbone_name == "resnet50":
        return resnet50
    elif backbone_name == "resnet101":
        return resnet101
    # elif backbone_name == "resnet50vd":
    #     return resnet50vd
    # elif backbone_name == "resnet101vd":
    #     return resnet101vd
    elif backbone_name == "mobilenetv2":
        pass
    raise RuntimeError(f"backbone should be resnet or mobilenet, not {backbone_name}.")


def build_feature_dim(backbone_name):
    """get feature dim
    """
    if "resnet" in backbone_name:
        return [2048, 1024, 512, 256]
    elif "mobile" in backbone_name:
        pass
    raise RuntimeError(f"backbone should be resnet or mobilenet, not {backbone_name}.")


def get_activation(activation):
    """activation function
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_activation_layer(activation):
    """activation layer
    """
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "gelu":
        return nn.GELU(inplace=True)
    elif activation == "glu":
        return nn.GLU(inplace=True)

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def conv_3x3_bn(inp_dim, oup_dim, padding=1, stride=1, bias=False, activation='relu'):
    """conv3x3 + bn + activate
    """
    return nn.Sequential(
            nn.Conv2d(inp_dim, oup_dim, kernel_size=3, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(oup_dim),
            get_activation_layer(activation)
        )


def conv_1x1_bn(inp_dim, oup_dim, padding=1, stride=1, bias=False, activation='relu'):
    """conv1x1 + bn + activate
    """
    return nn.Sequential(
            nn.Conv2d(inp_dim, oup_dim, kernel_size=1, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(oup_dim),
            get_activation_layer(activation)
        )


# Build the FPN layer
class FPN(nn.Module):
    def __init__(self, backbone_name, pretrain=True):
        super(FPN, self).__init__()
        self.pretrain = pretrain
        self.backbone_name = backbone_name
        # backbone
        self.backbone = build_backbone(self.backbone_name)()
        # feature dims
        self.features_dims = build_feature_dim(self.backbone_name)
        # FPN lat layer
        self.lat_layer_7 = conv_3x3_bn(self.features_dims[0], self.features_dims[3])        # (2048, 256)
        self.lat_layer_5 = conv_3x3_bn(self.features_dims[1], self.features_dims[3])        # (1024, 256)
        self.lat_layer_3 = conv_3x3_bn(self.features_dims[2], self.features_dims[3] // 2)   # (512 , 128)
        self.lat_layer_2 = conv_3x3_bn(self.features_dims[3], self.features_dims[3] // 4)   # (256 ,  64)

        # deconv channel
        self.deconv1 = conv_3x3_bn(self.features_dims[3], int(self.features_dims[3] / 2))
        self.deconv2 = conv_3x3_bn(int(self.features_dims[3] / 2), int(self.features_dims[3] / 4))

        # output features layer (64, 64)
        self.output_conv = nn.Conv2d(self.features_dims[3] // 4, self.features_dims[3] // 4, kernel_size=3, padding=1, stride=1, bias=False)

        if self.pretrain:
            self._initialize_weights()
            self._initialize_pretrain_models()
        else:
            self._initialize_weights()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(x)

    def _upsample_add(self, x, y):
        """upsampling add """
        _, _, H, W = y.size()
        return nn.Upsample(size=(H, W), mode='bilinear', align_corners=True)(x) + y

    def forward(self, x):
        features = self.backbone(x)
        c2, c3, c5, c7 = features
        p7 = self.lat_layer_7(c7)
        p5 = self._upsample_add(p7, self.lat_layer_5(c5))
        # reduce the channel from 256 to 128
        m5 = self.deconv1(p5)
        p3 = self._upsample_add(m5, self.lat_layer_3(c3))
        # reduce the channel from 128 to 64
        m3 = self.deconv2(p3)
        p2 = self._upsample_add(m3, self.lat_layer_2(c2))

        p1 = self.output_conv(p2)
        return p1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _load_models(self, weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        if 'model' in state_dict.keys():
            self.backbone.load_state_dict(state_dict['model'])
        elif 'state_dict' in state_dict.keys():
            self.backbone.load_state_dict(state_dict['state_dict'])
        else:
            self.backbone.load_state_dict(state_dict)

    def _initialize_pretrain_models(self):
        if self.backbone_name == "resnet50":
            self._load_models('/data/remote/github_code/OneFace/models/weights/resnet50.pth')
            print('Load the resnet50 backbone weights!!!!')
        elif self.backbone_name == "resnet101":
            self._load_models('weights/resnet101.pth')
            print('Load the resnet101 backbone weights!!!')
        elif self.backbone_name == "resnet50vd":
            pass
        elif self.backbone_name == "resnet101vd":
            pass
        elif self.backbone_name == "mobilenetv2":
            pass
        else:
            pass

# Build the Head layer
class Head(nn.Module):

    def __init__(self, num_classes=1, use_pts=True):
        super(Head, self).__init__()

        self.d_conv = 64
        self.bbox_dim = 4
        self.pts_dim = 10
        self.num_classes = num_classes
        self.use_pts = use_pts
        self.feats = nn.Conv2d(self.d_conv, self.d_conv, kernel_size=3, padding=1, stride=1, bias=False)
        # classification
        self.cls_score = nn.Conv2d(self.d_conv, self.num_classes, kernel_size=3, padding=1, stride=1, bias=False)
        # bbox
        self.ltrb_pred = nn.Conv2d(self.d_conv, self.bbox_dim, kernel_size=3, padding=1, stride=1, bias=False)
        # points
        self.pts_pred = nn.Conv2d(self.d_conv, self.pts_dim, kernel_size=3, padding=1, stride=1, bias=False)

        # Init parameters.
        prior_prob = 0.01
        self.bias_value = - math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def forward(self, features):
        """Return the classlogits, predbboxes or landmarks points
        """
        locations = self.locations(features)[None]
        feat = F.relu(self.feats(features))

        class_logits = self.cls_score(feat)
        pred_ltrb = F.relu(self.ltrb_pred(feat))
        pred_boxes = self.apply_ltrb(locations, pred_ltrb)
        if self.use_pts:
            # TODO: need fix the locations to x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 distance !!!
            pred_pts = self.pts_pred(feat)
            return class_logits, pred_boxes, pred_pts
        # The current version is not have the face points
        else:
            return class_logits, pred_boxes, None

    def apply_ltrb(self, locations, pred_ltrb):
        """retrun x1, y1, x2, y2
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W) (left, top, right, bottom)
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[:,0,:,:] = locations[:,0,:,:] - pred_ltrb[:,0,:,:]  # x1
        pred_boxes[:,1,:,:] = locations[:,1,:,:] - pred_ltrb[:,1,:,:]  # y1
        pred_boxes[:,2,:,:] = locations[:,0,:,:] + pred_ltrb[:,2,:,:]  # x2
        pred_boxes[:,3,:,:] = locations[:,1,:,:] + pred_ltrb[:,3,:,:]  # y2

        return pred_boxes

    @torch.no_grad()
    def locations(self, features, stride=4):
        """
        Build the locations for points anchor stride 4 for featuremap downstride
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device

        shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2

        locations = locations.reshape(h, w, 2).permute(2, 0, 1)

        return locations

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # initialize the bias for focal loss.
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, self.bias_value)

# Build the OneFace
class OneFace(nn.Module):
    def __init__(self, backbone_name, num_classes=1, pretrain=False, landmark=False):
        super(OneFace, self).__init__()

        self.backbone_name = backbone_name
        self.pretrain = pretrain
        self.num_classes = num_classes
        self.landmark = landmark
        self.fpn = FPN(self.backbone_name, self.pretrain)
        self.head = Head(self.num_classes, use_pts=self.landmark)

    def forward(self, x):
        features = self.fpn(x)
        cls_logits, pred_bbox, pred_pts = self.head(features)
        return cls_logits, pred_bbox, pred_pts

if __name__ == '__main__':
    model = OneFace(backbone_name="resnet101")
    print(model)
    input = torch.randn(1, 3, 800, 800)
    output = model(input)
    print(output[0].shape, output[1].shape, output)