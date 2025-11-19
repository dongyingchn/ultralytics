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

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect", "YOLOEDetect", "YOLOESegment", "Detect3D"


class Detect(nn.Module):
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

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

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
        cut_cls = self.conv_cut_cls(feat3d)                              #4*3*40*100
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

class Detect3D(Detect):
    """
    YOLO 3D Detection head:
    - 在标准 Detect 的基础上，新增一支 3D 回归分支 (cv4)，输出 extra_3d_dims 通道。
    - 通道顺序约定：前 9 维为基础 3D [x,y,z,l,w,h,rotation,xc2d,yc2d]，
      其后为 4 个面的信息（顺序：front, rear, left, right），每面 7 维 [x,y,z,xc2d,yc2d,score,is_vis]。
    - 训练: 返回 (x, extra3d)，其中 x 为 Detect 的多层原始输出（用于 2D loss），extra3d 为 3D 通道。
    - 推理导出(export=True): 直接将 extra3d 拼接到输出末尾，维度为 (bs, 4+nc+extra_3d_dims, N)。
    - 常规推理(export=False): 返回 (cat_out, (raw_levels, extra3d))，便于下游 postprocess/可视化与 3D 后处理。
    """

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """
        Args:
            nc (int): 类别数。
            ch (tuple): 各特征层通道数。
            extra_3d_dims (int): 3D 回归通道数量，默认 37 = 9 + 4*7。
        """
        super().__init__(nc, ch)
        self.n3d = None

        # 3D 分支：与 Pose/OBB 的实现风格一致
        # c4 = max(ch[0] // 4, self.n3d)
        self.cv4 = nn.ModuleList(
            head_group(c_in=x) for x in ch
        )

    def _extra3d_levels(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        逐层计算 3D 分支输出，返回列表：
        - 每层张量形状： (bs, n3d, H, W)
        """
        x_3d = [self.cv4[i](feats[i]) for i in range(self.nl)]
        self.n3d = x_3d[0].shape[1]

        return x_3d

    def _extra3d_cat(self, extra3d_levels: list[torch.Tensor]) -> torch.Tensor:
        """
        将逐层 3D 输出展平并拼接：
        返回形状：(bs, n3d, sum(H_i * W_i))
        """
        bs = extra3d_levels[0].shape[0]
        n3d = extra3d_levels[0].shape[1]
        return torch.cat([e.view(bs, n3d, -1) for e in extra3d_levels], dim=2)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        # 先基于原始特征 x[i] 计算 3D 分支（保持与 x 对齐的“按层”结构）
        extra3d_levels = self._extra3d_levels(x)  # list of (bs, n3d, Hi, Wi)

        # 再走 Detect 的 2D 路径
        x_out = Detect.forward(self, x)

        if self.training:
            # 训练态保持“按层”返回，x_out 是 list[levels]（Detect 的行为）
            # 同步返回 extra3d_levels，方便逐层用正样本索引 gather
            return x_out, extra3d_levels

        # 推理态：Detect.forward(export=False) 返回 (y_cat, raw_levels)，export=True 返回 y_cat
        if self.export:
            # 导出推理：把 3D 也展平 cat 后，直接拼接到 y 的通道末端
            extra3d_cat = self._extra3d_cat(extra3d_levels)  # (bs, n3d, N)
            return torch.cat([x_out, extra3d_cat], dim=1)
        else:
            # 常规推理：返回 (拼接后的y, (raw_levels, extra3d_cat))
            y_cat, raw_levels = x_out  # y_cat: (bs, 4+nc, N)
            extra3d_cat = self._extra3d_cat(extra3d_levels)  # (bs, n3d, N)
            return ((y_cat, extra3d_cat), (raw_levels, extra3d_levels))

    # Insert into your Detect3D / Detect head class
    def decode_extra3d(self, extra3d_cat, anchor_points=None, stride_tensor=None, as_numpy=False):
        """
        Decode Detect3D extra3d outputs into interpretable physical quantities.

        Accepts either:
        - extra3d_cat: Tensor shaped (B, n3d, N)
        - extra3d_cat: list/tuple of level tensors [(B, n3d, H, W), ...] -> will be flattened concat to (B, n3d, N)

        Returns dict with keys (torch tensors by default):
        - "raw": (B, N, n3d) raw vectors (per-anchor)
        - "base_raw": (B, N, base_dims)
        - "faces_raw": (B, N, num_faces, face_dims_out)
        - "base_decoded": dict of decoded base fields (x absent, we decode z, xc/yc as proj offsets to pixels if anchors provided,
                            sizes, rotation (heuristic), cutcls probs)
        - "faces_decoded": dict of decoded face fields (z, proj offsets or proj px, size, score prob)

        Assumptions (match loss and your provided layout):
        base_dims = 17 with layout:
            [ z3d, xc, yc, l3d, h3d, w3d, cls_logits(4), residuals(4), cutcls_logits(3) ]
        face_dims_out = 6 with layout per face:
            [ z3d_face, proj_x_off, proj_y_off, size_0, size_1, score_logit ]
        num_faces = 4

        Default inverse-normalization constants (tweak if your training hyperparams differ):
        z_scale = 40.0, z_offset = 40.0          (reverse (z-40)/40)
        size_scale = 18.0                        (reverse size/18)
        proj_scale = 6.0                         (network used tanh*6 in loss-level comparisons)
        face_size_scale = 18.0

        NOTE: rotation decoding from (cls_logits, residuals) is heuristic here: we compute class-center angles
        and add residuals (tanh-scaled). If your training used a precise angle-as-class encoding you should
        replace the heuristic with the exact inverse used during training.
        """
        import math
        import torch
        import numpy as np

        # Configurable attributes on head (fall back to defaults if not present)
        num_faces = getattr(self, "num_faces", 4)
        face_dims_out = getattr(self, "face_dims_out", 6)
        base_dims = getattr(self, "base_dims", 17)

        # decoding scales (defaults mirror loss.py usage; adapt if different)
        z_scale = getattr(self, "decode_z_scale", 40.0)
        z_offset = getattr(self, "decode_z_offset", 40.0)
        size_scale = getattr(self, "decode_size_scale", 18.0)
        proj_scale = getattr(self, "decode_proj_scale", 6.0)
        face_size_scale = getattr(self, "decode_face_size_scale", 18.0)

        # 1. Ensure tensor in (B, n3d, N) then (B, N, n3d)
        if isinstance(extra3d_cat, (list, tuple)):
            # flatten per-level (B, n3d, H, W) -> (B, n3d, -1) then concat
            try:
                bs = extra3d_cat[0].shape[0]
                n3d = extra3d_cat[0].shape[1]
                parts = [e.reshape(bs, n3d, -1) for e in extra3d_cat]
                extra3d = torch.cat(parts, dim=2)  # (B, n3d, N)
            except Exception:
                # fallback: try safe reshape
                tensors = []
                for e in extra3d_cat:
                    if e.ndim == 4:
                        tensors.append(e.reshape(e.shape[0], e.shape[1], -1))
                    elif e.ndim == 3:
                        tensors.append(e)
                    else:
                        raise RuntimeError("Unsupported extra3d level shape for decode_extra3d")
                extra3d = torch.cat(tensors, dim=2)
        else:
            extra3d = extra3d_cat  # assume (B, n3d, N)

        if not torch.is_tensor(extra3d):
            raise TypeError("extra3d_cat must be tensor or list/tuple of tensors")

        B, n3d, N = extra3d.shape
        raw_bn3d = extra3d.permute(0, 2, 1).contiguous()  # (B, N, n3d)

        # 2. Split into faces_raw and base_raw (if layout matches)
        expected_n3d = num_faces * face_dims_out + base_dims
        if n3d != expected_n3d:
            # shape mismatch: treat all remaining as base
            faces_raw = None
            base_raw = raw_bn3d  # (B,N,n3d)
        else:
            faces_raw = raw_bn3d[..., : num_faces * face_dims_out].reshape(B, N, num_faces, face_dims_out)
            base_raw = raw_bn3d[..., num_faces * face_dims_out :]  # (B, N, base_dims)

        # 3. Decode base_raw into interpretable fields
        base_decoded = {}
        if base_raw is not None:
            b = base_raw  # (B,N,base_dims)

            # z
            base_decoded["z3d"] = b[..., 0] * z_scale + z_offset

            # projection center offsets (xc, yc): network likely outputs offsets which were compared via tanh*proj_scale.
            # We'll apply tanh then scale. If during training you used another activation, change here.
            xc_off = torch.tanh(b[..., 1]) * proj_scale
            yc_off = torch.tanh(b[..., 2]) * proj_scale
            base_decoded["proj_offset_cell"] = torch.stack([xc_off, yc_off], dim=-1)  # (B,N,2)

            # sizes l,w,h (indices 3,4,5) -> decode
            base_decoded["l3d"] = b[..., 3] * size_scale
            base_decoded["h3d"] = b[..., 4] * size_scale
            base_decoded["w3d"] = b[..., 5] * size_scale

            # angle/class+residual: indices 6:10 (4 cls logits), 10:14 residuals
            ang_logits = b[..., 6:10]  # (B,N,4)
            ang_res = b[..., 10:14]    # (B,N,4)

            # convert logits -> probs
            try:
                ang_prob = torch.softmax(ang_logits, dim=-1)
            except Exception:
                ang_prob = ang_logits  # fallback

            
            # residuals are tanh-scaled roughly in (-1,1). Choose residual scale (rad) as pi/4 (tunable)
            res_scale = getattr(self, "decode_angle_residual_scale", (math.pi / 2.0))
            # expected angle = sum_k prob_k * (center_k + tanh(res_k)*res_scale)
            res_tanh = torch.tanh(ang_res) * res_scale  # (B,N,4)

            # # class-center angles used in training encoding (heuristic / mirror of loss.py):
            # # we use class-centers [0, pi/2, -pi/2, pi] as a reasonable inverse mapping;
            # # if your training used different centers replace accordingly.
            # centers = torch.tensor([0.0, math.pi / 2.0, -math.pi / 2.0, math.pi], device=b.device, dtype=b.dtype)
            # centers = centers.view(1, 1, 4)

            angle0 = res_tanh[..., 0]                         # 类0: 0 + res0
            angle1 = res_tanh[..., 1] + math.pi / 2.0        # 类1: +π/2 + res1
            angle2 = res_tanh[..., 2] - math.pi / 2.0        # 类2: -π/2 + res2
            res3  = res_tanh[..., 3]
            # 类3: 根据残差符号决定加/减 π，完全对齐方式一
            angle3 = torch.where(res3 > 0, res3 - math.pi, res3 + math.pi)

            angle_components = torch.stack([angle0, angle1, angle2, angle3], dim=-1)  # (B,N,4)
            # angle_components = centers + res_tanh  # (B,N,4)
            # weighted sum across classes
            # decoded_angle = (ang_prob * angle_components).sum(dim=-1)  # (B,N)
            index = torch.argmax(ang_prob, dim=-1)  # (B,N)
            decoded_angle = angle_components.gather(-1, index.unsqueeze(-1)).squeeze(-1)  # (B,N)
            base_decoded["rot_y"] = decoded_angle  # radians
            base_decoded["rot_y_cls"] = ang_prob  # (B,N,4)
            base_decoded["rot_y_res"] = res_tanh # (B,N,4)

            # cutcls logits (3 classes) indices 14:17 -> probabilities
            cutcls_logits = b[..., 14:17] if base_dims >= 17 else None
            if cutcls_logits is not None:
                try:
                    base_decoded["cutcls_prob"] = torch.sigmoid(cutcls_logits)
                except Exception:
                    base_decoded["cutcls_prob"] = cutcls_logits

            # Optionally convert proj offsets to pixel coordinates if anchor_points and stride_tensor available
            # anchor_points expected shape: (N,2) and stride_tensor shape (N,) or (N,1)
            
            # anchor_points shape is actually (2, N) and stride_tensor (N,).
            if anchor_points is None or stride_tensor is None:
                # try to fallback to attributes on head (some implementations store anchors/strides)
                
                anchor_points = (
                    anchors if (anchors := getattr(self, "anchors", None)) is not None
                    else anchor_points
                )
                stride_tensor = (
                    strides if (strides := getattr(self, "strides", None)) is not None
                    else strides
                )
                
                # reshape if needed
                if anchor_points is not None and isinstance(anchor_points, torch.Tensor):
                    if anchor_points.ndim == 2 and anchor_points.shape[0] == 2:
                        anchor_points = anchor_points.transpose(0, 1)
                if stride_tensor is not None and isinstance(stride_tensor, torch.Tensor):
                    if stride_tensor.ndim == 1:
                        pass
                    elif stride_tensor.ndim == 2 and stride_tensor.shape[0] == 1:
                        stride_tensor = stride_tensor.view(-1)

            if anchor_points is not None and stride_tensor is not None:
                ap = anchor_points.to(device=b.device, dtype=b.dtype)
                st = stride_tensor.to(device=b.device, dtype=b.dtype)
                # reshape ap to (1,N,2), st to (1,N,1)
                if ap.ndim == 3:
                    ap = ap.reshape(-1, 2)
                ap_t = ap.view(1, -1, 2)
                st_t = st.view(1, -1, 1)
                proj_px = (ap_t + base_decoded["proj_offset_cell"]) * st_t  # (B,N,2)
                base_decoded["proj_px"] = proj_px
            else:
                # leave offsets in cell units
                base_decoded["proj_px"] = None

        # 4. Decode faces
        faces_decoded = None
        if faces_raw is not None:
            fr = faces_raw  # (B,N,num_faces,face_dims_out)
            # z face
            fz = fr[..., 0] * z_scale + z_offset
            # proj offsets per-face, apply tanh*proj_scale as in loss usage
            fproj_off_x = torch.tanh(fr[..., 1]) * proj_scale
            fproj_off_y = torch.tanh(fr[..., 2]) * proj_scale
            fproj_off = torch.stack([fproj_off_x, fproj_off_y], dim=-1)  # (B,N,num_faces,2)
            # sizes (2 dims)
            fsize = fr[..., 3:5] * face_size_scale
            # score probability
            fscore = fr[..., 5] if fr.shape[-1] >= 6 else None

            faces_decoded = {
                "z3d": fz,
                "proj_offset_cell": fproj_off,
                "size": fsize,
                "score_prob": fscore,
            }

            # convert to pixel coords if anchors/strides provided
            if anchor_points is not None and stride_tensor is not None:
                ap = anchor_points.to(device=fr.device, dtype=fr.dtype)
                st = stride_tensor.to(device=fr.device, dtype=fr.dtype)
                if ap.ndim == 3:
                    ap = ap.reshape(-1, 2)
                ap_t = ap.view(1, -1, 2).unsqueeze(2)  # (1, N, 1, 2)
                st_t = st.view(1, -1, 1).unsqueeze(2)  # (1, N, 1, 1)
                proj_px_face = (ap_t + fproj_off) * st_t  # (B, N, num_faces, 2)
                faces_decoded["proj_px"] = proj_px_face

        # 5. Prepare outputs, optionally convert to numpy
        def maybe_numpy(x):
            if x is None:
                return None
            return x.detach().cpu().numpy() if as_numpy and isinstance(x, torch.Tensor) else x

        out = {
            "raw": raw_bn3d,  # torch tensor (B, N, n3d)
            "faces_raw": faces_raw,  # torch tensor or None
            "base_raw": base_raw,  # torch tensor or None
            "base_decoded": {k: maybe_numpy(v) for k, v in base_decoded.items()} if base_decoded else {},
            "faces_decoded": {k: maybe_numpy(v) for k, v in faces_decoded.items()} if faces_decoded else {},
        }
        return out

    @staticmethod
    def split_extra(extra3d: torch.Tensor, base_dims: int = 9, face_dims: int = 7, num_faces: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将 extra3d 拆分为基础 3D 与面信息（不更改排列，仅做通道切分）。
        Args:
            extra3d: (bs, n3d, N) 3D 通道输出
            base_dims: 基础 3D 通道数，默认 9
            face_dims: 单个面通道数，默认 7
            num_faces: 面数量，默认 4 (front, rear, left, right)

        Returns:
            base3d: (bs, 9, N)
            faces:  (bs, 4*7, N)  按顺序拼接的 4 个面
        """
        base3d = extra3d[:, 0:base_dims, :]
        faces = extra3d[:, base_dims : base_dims + num_faces * face_dims, :]
        return base3d, faces

class Segment(Detect):
    """
    YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, ch: tuple = ()):
        """
        Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """
    YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, ch: tuple = ()):
        """
        Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """
    YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), ch: tuple = ()):
        """
        Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """
    YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass of the YOLO model on input image data.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """
        Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int, optional): Kernel size.
            s (int, optional): Stride.
            p (int, optional): Padding.
            g (int, optional): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """
    Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding
    in object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: tuple = ()):
        """
        Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """
    Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with
    dynamic vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """
        Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    def conv2linear(self, conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2)), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2)), torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool
            )


class YOLOEDetect(Detect):
    """
    Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: tuple = ()):
        """
        Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )

        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor):
        """Fuse text features with model weights for efficient inference."""
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        for cls_head, bn_head in zip(self.cv3, self.cv4):
            assert isinstance(cls_head, nn.Sequential)
            assert isinstance(bn_head, BNContrastiveHead)
            conv = cls_head[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_head.logit_scale
            bias = bn_head.bias
            norm = bn_head.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_head[-1] = conv

            bn_head.fuse()

        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward_lrpc(self, x: list[torch.Tensor], return_mask: bool = False) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        masks = []
        assert self.is_fused, "Prompt-free inference requires model to be fused!"
        for i in range(self.nl):
            cls_feat = self.cv3[i](x[i])
            loc_feat = self.cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            x[i], mask = self.lrpc[i](
                cls_feat, loc_feat, 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001)
            )
            masks.append(mask)
        shape = x[0][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors([b[0] for b in x], self.stride, 0.5))
            self.shape = shape
        box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
        cls = torch.cat([xi[1] for xi in x], 2)

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

        mask = torch.cat(masks)
        y = torch.cat((dbox if self.export and not self.dynamic else dbox[..., mask], cls.sigmoid()), 1)

        if return_mask:
            return (y, mask) if self.export else ((y, x), mask)
        else:
            return y if self.export else (y, x)

    def forward(self, x: list[torch.Tensor], cls_pe: torch.Tensor, return_mask: bool = False) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x, return_mask)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), cls_pe)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize biases for detection heads."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, c, s in zip(m.cv2, m.cv3, m.cv4, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLOESegment(YOLOEDetect):
    """
    YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks
    with text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self, nc: int = 80, nm: int = 32, npr: int = 256, embed: int = 512, with_bn: bool = False, ch: tuple = ()
    ):
        """
        Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> tuple | torch.Tensor:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        has_lrpc = hasattr(self, "lrpc")

        if not has_lrpc:
            x = YOLOEDetect.forward(self, x, text)
        else:
            x, mask = YOLOEDetect.forward(self, x, text, return_mask=True)

        if self.training:
            return x, mc, p

        if has_lrpc:
            mc = (mc * mask.int()) if self.export and not self.dynamic else mc[..., mask]

        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """
        Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """
        Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, 300, 4+nc) containing bounding boxes and
                class scores.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(
        self,
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """
        Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions
    for improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """
        Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = nn.ModuleList([nn.Identity()] * self.nl)
