# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect_with_3D"

class head_group(nn.Module):
    def __init__(self, c_in, n_anchor=None, mergeBn=False):
        super(head_group, self).__init__()
        self.n_anchor = n_anchor

        # need to be readin as parameters for different cameras
        # sx = 800.0 / 3840.0
        # sy = 320.0 / 1536.0
        # self.cam_intrinsics = nn.Parameter(torch.tensor([2459.49*sx/100, 
        #                                                  1920.62*sx/100, 
        #                                                  2456.18*sy/100, 
        #                                                  1086.58*sy/100, 
        #                                                  -0.33891, 
        #                                                  0.203, 
        #                                                  -0.137994, 
        #                                                  0.0447447], dtype=torch.float32)) #fu、cu、fv、cv、distort_coeffs
        # self.cam_intrinsics.requires_grad = False
        '''
        For each face, output:
            z3d: the 3d position of the center of this face
            projected 3d center: u, v
            size2d: 2 measurable dim of 'l', 'h', 'w'
            face confidence: is this face visible from camera
            Dim = 1 + 2 + 2 + 1 = 6
        
        To predict the whole 3D box of certain object such as Pedestrian at once:
            z3d: the center of the 3D box
            projected 3d center: u, v
            lhw: the size of the 3D box
            Dim = 1 + 2 + 3 = 6
        
        To predict heading:
            yaw: the heading angle of the 3D box, predict sin(x) & cos(x), learning based on faces
            Dim=2
        '''
        c_out = c_in
        if mergeBn:
            self.conv_3d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
        
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1),
         
            nn.ReLU()
                        )
        else:
            self.conv_3d = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, 1, 1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, 1, 1),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )

        self.conv_front_feat  = nn.Sequential(nn.Conv2d(c_out, c_out, 1, 1, 0), 
                                              nn.ReLU())
        self.conv_front_face  = nn.Conv2d(c_out, 5, 1, 1, 0)
        self.conv_front_score = nn.Conv2d(c_out, 1, 1, 1, 0)

        self.conv_tail_feat  = nn.Sequential(nn.Conv2d(c_out, c_out, 1, 1, 0), 
                                              nn.ReLU())
        self.conv_tail_face  = nn.Conv2d(c_out, 5, 1, 1, 0)
        self.conv_tail_score = nn.Conv2d(c_out, 1, 1, 1, 0)

        self.conv_left_feat  = nn.Sequential(nn.Conv2d(c_out, c_out, 1, 1, 0), 
                                              nn.ReLU())
        self.conv_left_face  = nn.Conv2d(c_out, 5, 1, 1, 0)
        self.conv_left_score = nn.Conv2d(c_out, 1, 1, 1, 0)

        self.conv_right_feat  = nn.Sequential(nn.Conv2d(c_out, c_out, 1, 1, 0), 
                                              nn.ReLU())
        self.conv_right_face  = nn.Conv2d(c_out, 5, 1, 1, 0)
        self.conv_right_score = nn.Conv2d(c_out, 1, 1, 1, 0)
        
        self.conv_whole   = nn.Sequential(nn.Conv2d(c_out, c_out, 1, 1, 0),
                                          nn.ReLU(),
                                          nn.Conv2d(c_out, 6, 1, 1, 0))
        
        # self.conv_heading_feat = nn.Sequential(nn.Conv2d(c_out+5*4, 256, 1, 1, 0),
        #                                        nn.BatchNorm2d(256),
        #                                        nn.ReLU(),
        #                                        nn.Conv2d(256, 256, 1, 1, 0),
        #                                        nn.BatchNorm2d(256),
        #                                        nn.ReLU(),
        #                                        nn.Conv2d(256, 256, 1, 1, 0),
        #                                        nn.BatchNorm2d(256),
        #                                        nn.ReLU())
        if mergeBn:
            self.conv_heading_feat = nn.Sequential(nn.Conv2d(c_out, 256, 1, 1, 0),
                          
                                            nn.ReLU(),
                                            nn.Conv2d(256, 256, 1, 1, 0),
                 
                                            nn.ReLU(),
                                            nn.Conv2d(256, 256, 1, 1, 0),
                           
                                            nn.ReLU())
        else:
            self.conv_heading_feat = nn.Sequential(nn.Conv2d(c_out, 256, 1, 1, 0),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU(),
                                                nn.Conv2d(256, 256, 1, 1, 0),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU(),
                                                nn.Conv2d(256, 256, 1, 1, 0),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU())
        self.conv_heading = nn.Conv2d(256, 8, 1, 1, 0)

        self.conv_cut_cls = nn.Sequential(nn.Conv2d(c_out, c_out, 1, 1, 0),
                                          nn.ReLU(),
                                          nn.Conv2d(c_out, 3, 1, 1, 0))
        # self.conv_plate = nn.Sequential(nn.Conv2d(c_out, c_out, 1, 1, 0),
        #                                   nn.ReLU(),
        #                                   nn.Conv2d(c_out, 1, 1, 1, 0))

        # initialize params
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if name == 'conv_heading':
                    m.bias.data[0] = 0.0
                    m.bias.data[1] = 0.0
                    m.bias.data[2] = 1.0
                    m.bias.data[3] = 0.0
        
    def forward(self, x):
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        feat3d = self.conv_3d(x)                #4*256*40*100
        featF = self.conv_front_feat(feat3d)
        featT = self.conv_tail_feat(feat3d)
        featL = self.conv_left_feat(feat3d)
        featR = self.conv_right_feat(feat3d)
        whole = self.conv_whole(feat3d)         #4*6*40*100

        outF = self.conv_front_face(featF)      #4*5*40*100
        outT = self.conv_tail_face(featT)
        outL = self.conv_left_face(featL)
        outR = self.conv_right_face(featR)

        scoreF = self.conv_front_score(featF)   #4*1*40*100
        scoreT = self.conv_tail_score(featT)
        scoreL = self.conv_left_score(featL)
        scoreR = self.conv_right_score(featR)
        
        # concat heading feat
        # cam_params = self.cam_intrinsics.view(1, 8, 1, 1).repeat(bz, 1, h, w)    #4*8*40*100
        # heading_feat = torch.cat([feat3d,
        #                           outF*scoreF, 
        #                           outT*scoreT, 
        #                           outL*scoreL, 
        #                           outR*scoreR
                          
        #                           ], dim=1)     #4*284*40*100
        heading = self.conv_heading(self.conv_heading_feat(feat3d))      #4*8*40*100
        cut_cls = self.conv_cut_cls(feat3d)
        # plate = self.conv_plate(feat3d)
        out  = torch.cat([outF,
                          scoreF,
                          outT,
                          scoreT,
                          outL,
                          scoreL,
                          outR,
                          scoreR,
                          whole,
                          heading,
                          cut_cls], dim=1)
        return out

class Detect_with_3D(nn.Module):
    """
    YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        forward_end2end: Perform forward pass for end-to-end detection.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """
        Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        self.cv4 = nn.ModuleList(head_group(c_in=x) for x in ch)

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)
        
        # if self.end2end:
        #     return self.forward_end2end(x)

        # for i in range(self.nl):
        #     x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # if self.training:  # Training path
        #     return x
        # y = self._inference(x)
        # return y if self.export else (y, x)

        x_2d = []
        x_3d = []
        for i in range(self.nl):
            x_2d.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
            x_3d.append(self.cv4[i](x[i]))
        if self.training:  # Training path
            return x_2d, x_3d
        y = self._inference(x_2d)
        return y if self.export else (y, x_2d)

    def forward_end2end(self, x: list[torch.Tensor]) -> dict | tuple:
        """
        Perform forward pass of the v10Detect module.

        Args:
            x (list[torch.Tensor]): Input feature maps from different levels.

        Returns:
            outputs (dict | tuple): Training mode returns dict with one2many and one2one outputs.
                Inference mode returns processed detections or tuple with detections and raw outputs.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (list[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        """
        Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


