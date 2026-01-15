import cv2
from ultralytics import YOLO

import os
import glob
import json
import numpy as np
import math
from tqdm import tqdm
import random
import argparse

random.seed(42)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++ Functions ported from eval_3D_roi +++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    rec = np.array(rec)
    prec = np.array(prec)
    if use_07_metric:  # Using 2007 method
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # Interpolate
            ap = ap + p / 11.
    else:  # New method, calculate all points
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def iou_func(box1, box2):
    box = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    if box[0] >= box[2] or box[1] >= box[3]:
        return 0
    else:
        area = (box[2] - box[0]) * (box[3] - box[1])
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return area / float(area1 + area2 - area + 1e-6)

def eval_tp_fp_3drelategap(num_class, gtresult, dtresult, gtcount):
    aps = []
    TP, precision, recall, dtcount, conf = {}, {}, {}, {}, {}
    obj_3d, obj_x3d, obj_depth, obj_heading = {}, {}, {}, {}
    x3d_distance, depth_distance, heading_distance, num_distance = {}, {}, {}, {}
    lx3d, ldepth, lheading, dis_x3d, dis_depth, dis_heading = {}, {}, {}, {}, {}, {}

    face_type_index = {'front': 0, 'tail': 1, 'left': 2, 'right': 3, 'whole': -1}

    for id in range(num_class):
        if id not in dtresult:
            aps.append(0.0)
            continue
        
        dtcount[id], TP[id] = 0, 0
        precision[id], recall[id], conf[id] = [], [], []
        obj_3d[id], obj_x3d[id], obj_depth[id], obj_heading[id] = 0, 0, 0, 0
        lx3d[id], ldepth[id], dis_x3d[id], dis_depth[id], dis_heading[id] = [], [], [], [], []
        lheading[id] = []
        x3d_distance[id], depth_distance[id], heading_distance[id], num_distance[id] = [0] * 10, [0] * 10, [0] * 10, [0] * 10

        dtresult[id].sort(key=lambda x: x[1], reverse=True) # sort by confidence
        
        for i in range(len(dtresult[id])):
            (name, confidence, BB, pred3d_parsed, _, _, _, _) = dtresult[id][i]
            dtcount[id] += 1
            
            img_name_key = os.path.basename(name).split('.')[0]
            if img_name_key in gtresult:
                for (BBgt, BBcls, BB3d) in gtresult[img_name_key]:
                    if iou_func(BB, BBgt) > 0.5:
                        if int(id) == int(BBcls):
                            TP[id] += 1
                            # 3D evaluation for vehicle categories
                            if id > 9 and id <= 16 and BB3d[8] != -1 and abs(BB3d[7]) <= 60 and abs(BB3d[5]) <= 16:
                                infer_x3d, infer_y3d, z, l, h, w, yaw, face_type, cut_cls = pred3d_parsed
                                
                                gt_x, gt_y, gt_z = BB3d[5], BB3d[6], BB3d[7]
                                pred_x, pred_y, pred_z = infer_x3d, infer_y3d, z
                                
                                gt_yaw = BB3d[11]
                                pred_yaw = yaw
                                
                                depth_gap = abs(gt_z - pred_z) / gt_z if gt_z > 0 else 0
                                x3d_gap = abs(gt_x - pred_x)
                                heading_gap = abs(gt_yaw - pred_yaw)
                                heading_gap = min(2 * np.pi - heading_gap, heading_gap)

                                obj_depth[id] += depth_gap
                                obj_heading[id] += heading_gap
                                obj_x3d[id] += x3d_gap
                                obj_3d[id] += 1
                                
                                distance_seq = int(min(int(gt_z / 10.0), 9))
                                x3d_distance[id][distance_seq] += x3d_gap
                                depth_distance[id][distance_seq] += depth_gap
                                heading_distance[id][distance_seq] += heading_gap
                                num_distance[id][distance_seq] += 1
                            elif id <= 9 and BB3d[8] != -1 and abs(BB3d[7]) <= 60 and abs(BB3d[5]) <= 16:
                                infer_x3d, infer_y3d, z, l, h, w, yaw, face_type, cut_cls = pred3d_parsed

                                index = face_type_index[face_type]

                                # scores_gt = np.array([BB3d[24],BB3d[32],BB3d[40],BB3d[48]])
                                # face_cls_gt = np.argmax(scores_gt)
                                gt_3d = [BB3d[18+index*8 + 0],BB3d[18+index*8 + 1],BB3d[18+index*8 + 2],BB3d[8],BB3d[9],BB3d[10],BB3d[11]]
                                # corners = cam_corners_front_rear(np.array([infer_x3d, infer_y3d, z, l, h, w, yaw]), face_type)
                                
                                gt_x, gt_y, gt_z = gt_3d[0], gt_3d[1], gt_3d[2]
                                pred_x, pred_y, pred_z = infer_x3d, infer_y3d, z
                                
                                gt_yaw = BB3d[11]
                                pred_yaw = yaw
                                
                                depth_gap = abs(gt_z - pred_z) / gt_z if gt_z > 0 else 0
                                x3d_gap = abs(gt_x - pred_x)
                                heading_gap = abs(gt_yaw - pred_yaw)
                                heading_gap = min(2 * np.pi - heading_gap, heading_gap)

                                obj_depth[id] += depth_gap
                                obj_heading[id] += heading_gap
                                obj_x3d[id] += x3d_gap
                                obj_3d[id] += 1
                                
                                distance_seq = int(min(int(gt_z / 10.0), 9))
                                x3d_distance[id][distance_seq] += x3d_gap
                                depth_distance[id][distance_seq] += depth_gap
                                heading_distance[id][distance_seq] += heading_gap
                                num_distance[id][distance_seq] += 1
                            break
            
            conf[id].append(confidence)
            precision[id].append(float(TP[id]) / (float(dtcount[id] + 1e-6)))
            recall[id].append(float(TP[id]) / (float(gtcount.get(id, 0) + 1e-6)))

            lx3d[id].append(float(obj_x3d.get(id, 0)) / (float(obj_3d.get(id, 0) + 1e-6)))
            ldepth[id].append(float(obj_depth.get(id, 0)) / (float(obj_3d.get(id, 0) + 1e-6)))
            lheading[id].append(float(obj_heading.get(id, 0)) / (float(obj_3d.get(id, 0) + 1e-6)))

        if not dtresult[id]:
            ap = 0.0
        else:
            ap = voc_ap(recall[id], precision[id], use_07_metric=True)
        aps.append(ap)

        for i in range(10):
            dis_x3d[id].append(float(x3d_distance[id][i]) / (float(num_distance[id][i]) + 1e-6))
            dis_depth[id].append(float(depth_distance[id][i]) / (float(num_distance[id][i]) + 1e-6))
            dis_heading[id].append(float(heading_distance[id][i]) / (float(num_distance[id][i]) + 1e-6))

    return precision, recall, aps, conf, lx3d, ldepth, lheading, dis_x3d, dis_depth, dis_heading, num_distance

def gttransfor3d(textlists, class_names_str2id, roi):
    gtresult = {}
    gtcount = {}
    # textlists = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"Found {len(textlists)} label files.")
    for text in tqdm(textlists, desc="Loading GT labels"):
        filename = os.path.basename(text)[:-4]
        gtresult[filename] = []
        with open(text, 'r') as f:
            labellines = f.readlines()
        
        for line in labellines:
            data = line.strip().split(' ')
            cls_id = class_names_str2id[data[0]]  # Changed from function call to dictionary access
            
            # Assuming image size is 3840x2160 for YOLO coord conversion
            # TODO: Make image size dynamic if needed
            w_img, h_img = 3840, 2160
            x_center, y_center, w, h = [float(v) for v in data[1:5]]
            x1 = (x_center - w / 2) * w_img
            y1 = (y_center - h / 2) * h_img
            x2 = (x_center + w / 2) * w_img
            y2 = (y_center + h / 2) * h_img

            # --- ROI Clipping Logic ---
            if roi is not None:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi
                x1 = max(x1, roi_x1)
                y1 = max(y1, roi_y1)
                x2 = min(x2, roi_x2)
                y2 = min(y2, roi_y2)

            box = [x1, y1, x2, y2]

            h_box = y2 - y1
            w_box = x2 - x1
            if h_box < 40 or w_box < 40:
                continue

            gtcount[cls_id] = gtcount.get(cls_id, 0) + 1
            
            gt3d = np.ones(50) * -1
            if len(data) > 5 and data[5] != '-1':
                gt3d[:len(data)] = [class_names_str2id[data[0]]] + [float(v) for v in data[1:]]

            gtresult[filename].append((box, cls_id, gt3d))
    return gtresult, gtcount

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++ Functions from original predict_3D.py +++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_rot_mat_and_trans(calib_dict):
    su = np.sin(calib_dict['roll'] * np.pi / 180)
    cu = np.cos(calib_dict['roll'] * np.pi / 180)
    sv = np.sin(calib_dict['pitch'] * np.pi / 180)
    cv = np.cos(calib_dict['pitch'] * np.pi / 180)
    sw = np.sin(calib_dict['yaw'] * np.pi / 180)
    cw = np.cos(calib_dict['yaw'] * np.pi / 180)

    rot = np.array([[cv * cw, su * sv * cw - cu * sw, su * sw + cu * sv * cw],
                    [cv * sw, cu * cw + su * sv * sw, cu * sv * sw - su * cw],
                    [-sv, su * cv, cu * cv]])
    rot = np.transpose(rot)

    rot_yxz = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    rot_ego2cam = np.matmul(rot_yxz, rot)
    
    pos = np.array(calib_dict['pos'])
    trans_ego2cam = -1 * np.matmul(rot_ego2cam, pos)

    ego_to_cam = np.zeros((4, 4))
    ego_to_cam[:3, :3] = rot_ego2cam
    ego_to_cam[:3, 3] = trans_ego2cam
    ego_to_cam[3, 3] = 1

    return ego_to_cam

def read_calibs(calib_path):
    if isinstance(calib_path, dict):
        calib_info = calib_path
    elif isinstance(calib_path, str) and os.path.isfile(calib_path):
        with open(calib_path, 'r') as f:
            calib_info = json.load(f)
    else:
        raise ValueError("calib_path must be a file path or a dict")

    intrinsic = np.array([
        [calib_info['focal_u'], 0, calib_info['cu']],
        [0, calib_info['focal_v'], calib_info['cv']],
        [0, 0, 1]
    ], dtype=np.float64)
    extrinsic = get_rot_mat_and_trans(calib_info)
    distortion = np.array(calib_info['distort_coeffs'])

    # Store vehicle pose for BEV transformation
    vehicle_pose = {
        'pos': calib_info['pos'],
        'pitch': calib_info['pitch'],
        'yaw': calib_info['yaw'],
        'roll': calib_info['roll']
    }

    return intrinsic, extrinsic, distortion, vehicle_pose

def rotation_3d_in_axis(points, angles, axis=1):
    # This version is for batch rotation, keep for compatibility if needed.
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1: # Y-axis
        rot_mat_T = np.stack([
            np.stack([rot_cos, zeros, -rot_sin]),
            np.stack([zeros, ones, zeros]),
            np.stack([rot_sin, zeros, rot_cos])
        ])
    else:
        raise ValueError(f'axis {axis} not supported for batch rotation')

    return np.einsum('ij, jk', points, rot_mat_T)

def cam_corners_front_rear(pred3d, facetype):
    dims = pred3d[3:6]
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    
    origins = {
        'tail': [0, 0.5, 0.5], 'front': [1, 0.5, 0.5],
        'right': [0.5, 0.5, 0], 'left': [0.5, 0.5, 1],
        'whole': [0.5, 0.5, 0.5]
    }
    if facetype not in origins:
        raise AssertionError('Non valid face type')
    
    corners_norm = corners_norm - origins[facetype]
    corners = dims.reshape([1, 3]) * corners_norm.reshape([8, 3])
    corners = rotation_3d_in_axis(corners, pred3d[6], axis=1)
    corners += pred3d[:3].reshape(1, 3)
    return corners

def img_cam_kb(img_corners, cam_intrinsic, distort_param):
    ones_col = np.ones((img_corners.shape[0], 1), dtype=np.float32)
    img_corners_hom = np.concatenate((img_corners.astype(np.float32), ones_col), axis=-1)
    
    k1, k2, k3, k4 = distort_param[:4]
    
    points = np.dot(np.linalg.inv(cam_intrinsic), img_corners_hom.T).T
    xp = points[:, 0]
    yp = points[:, 1]
    thetaD = np.sqrt(xp * xp + yp * yp)
    theta = thetaD.copy()

    for _ in range(20):
        theta_2 = theta**2
        theta_4 = theta_2**2
        theta_6 = theta_4*theta_2
        theta_8 = theta_4**2
        ftheta = theta * (1 + k1*theta_2 + k2*theta_4 + k3*theta_6 + k4*theta_8) - thetaD
        ftheta_dao = (1 + 3*k1*theta_2 + 5*k2*theta_4 + 7*k3*theta_6 + 9*k4*theta_8)
        theta_update = ftheta / ftheta_dao
        theta -= theta_update
        if np.all(np.abs(theta_update) < 1e-6):
            break
            
    r = np.tan(theta)
    
    mask = thetaD > 1e-6
    normx = np.ones_like(xp)
    normy = np.ones_like(yp)
    normx[mask] = xp[mask] * r[mask] / thetaD[mask]
    normy[mask] = yp[mask] * r[mask] / thetaD[mask]

    return normx[0], normy[0]

def cam_img_kb(x, y, z, cam_intrinsic, distort_param):
    if z < 1e-6: return np.array([-1, -1])
    
    cam_x, cam_y = x/z, y/z
    r = np.sqrt(cam_x**2 + cam_y**2)
    theta = np.arctan(r)
    
    k1, k2, k3, k4 = distort_param[:4]
    theta_2 = theta**2
    theta_4 = theta_2**2
    theta_6 = theta_4*theta_2
    theta_8 = theta_4**2
    
    thetaD = theta * (1 + k1*theta_2 + k2*theta_4 + k3*theta_6 + k4*theta_8)
    
    scale = thetaD / r if r > 1e-6 else 1.0
    xp, yp = cam_x * scale, cam_y * scale
    
    fx, fy = cam_intrinsic[0, 0], cam_intrinsic[1, 1]
    cx, cy = cam_intrinsic[0, 2], cam_intrinsic[1, 2]
    
    imgx = fx * xp + cx
    imgy = fy * yp + cy

    return np.array([imgx, imgy])

def drawPointBox(img, imgW, imgH, rect_corners, colors, thickness=1):

    line_indices = ((4, 5), (5, 6), (6, 7), (7, 4),       # 尾部
                    (0,4), (1,5), (2,6),(3,7),
                    (0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3))     # 头部(1, 3),  
    
    border_x = imgW ####
    border_y = imgH #####
    for i in range(len(rect_corners)):
        if int(rect_corners[i][0]) < 0 or int(rect_corners[i][0]) > border_x or int(rect_corners[i][1]) < 0 or int(rect_corners[i][1]) > border_y:
            continue    
        # cv2.circle(img,(int(rect_corners[i][0]),int(rect_corners[i][1])),5,(255,0,0),2,cv2.LINE_8,0)
        # cv2.putText(img, str(i), (int(rect_corners[i][0]+5),int(rect_corners[i][1] - 5)), 1, 1, (0, 0, 255), lineType=cv2.LINE_AA)

    corners = rect_corners.astype(np.int32)
    for start, end in line_indices:
        try:
            if (start,end) in [(0, 1), (1, 2), (2, 3), (3, 0),(0, 2), (1, 3)] and corners.shape[0]==8:
                if int(corners[start, 0]) < 5 or int(corners[start, 0]) > (border_x - 5) or int(corners[start, 1]) < 5 or int(corners[start, 1]) > (border_y - 5):
                    continue 
                if int(corners[end, 0]) < 5 or int(corners[end, 0]) > (border_x - 5) or int(corners[end, 1]) < 5 or int(corners[end, 1]) > (border_y - 5):
                    continue
                cv2.line(img, (corners[start, 0], corners[start, 1]),(corners[end, 0], corners[end, 1]), colors[1], thickness*2,cv2.LINE_AA)
            else:
                cv2.line(img, (corners[start, 0], corners[start, 1]),(corners[end, 0], corners[end, 1]), colors[0], thickness,cv2.LINE_AA)
        except:
            pass
    return img

def yuv444_bt601_full_range2rgb(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[1.000, 0.000, 1.402],[1.000, -0.344136, -0.714136],[1.000, 1.772, 0.000]])
    bias = np.array([0, 128, 128])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq-bias_seq),0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++ BEV Functions  +++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def draw_dashed_line(img, start_point, end_point, color, thickness, dash_length, space_length):
    x1, y1 = start_point
    x2, y2 = end_point
    dx = x2 - x1
    dy = y2 - y1
    length = int(np.sqrt(dx * dx + dy * dy))
    dash_count = int(length / (dash_length + space_length))

    for i in range(dash_count):
        start = i * (dash_length + space_length)
        end = start + dash_length
        x_start = int(x1 + start * dx / length)
        y_start = int(y1 + start * dy / length)
        x_end = int(x1 + end * dx / length)
        y_end = int(y1 + end * dy / length)
        cv2.line(img, (x_start, y_start), (x_end, y_end), color, thickness)
    
def drawbev(bevimg, vehicle3d, face_type, is_pred=True):
    # x,y,z,l,w,h = vehicle3d[0], -vehicle3d[1],vehicle3d[2],vehicle3d[3],vehicle3d[4],vehicle3d[5] ###北西天，y取反对应
    x,y,z,l,w,h = vehicle3d[0], vehicle3d[1],vehicle3d[2],vehicle3d[3],vehicle3d[4],vehicle3d[5] ###右下前，y取反对应
    rotation_y = vehicle3d[-1]
    yaw = rotation_y
    # if yaw < 0:
    #     yaw = yaw + math.pi * 2
    radians = yaw

    # 目标图像参数
    W_m = 100  # 目标图像宽度（米）
    H_m = 100  # 目标图像高度（米）
    dx = 0.1  # 每个像素对应的宽度（米 ）
    dy = 0.1  # 每个像素对应的高度（米）
    if  x > 50.0 or x < -50.0 or z > 100.0:
        return bevimg

    corners = cam_corners_front_rear(np.array([x, y, z, l, h, w, rotation_y]), face_type)
    xyz3d_0 =  np.mean(corners[4:8,:],axis=0) ### Front_point
    xyz3d_1 = np.mean(corners[0:4,:],axis=0) ### Back_point
    xyz3d_2 = np.mean((corners[1,:],corners[2,:],corners[5,:],corners[6,:]),axis=0) ### Left_point
    xyz3d_3 = np.mean((corners[0,:],corners[4,:],corners[3,:],corners[7,:]),axis=0) ### Right_point
    xyz3d_center = np.mean(corners[0:8,:],axis=0)  ###center

    center = [int(500 + xyz3d_center[0] / dx),  1000 - int(xyz3d_center[2] / dy)]
    front_point = [int(500 + xyz3d_0[0] / dx),  1000 - int(xyz3d_0[2] / dy)]
    
    H,W = l / dy, w / dx
    size = [H, W]
    angle = np.degrees(rotation_y) 
    rect = (center, size, angle)  # 创建旋转矩形的参数结构
    box = cv2.boxPoints(rect)    # 获取四个顶点坐标
    box = np.intp(box)           # 转换为整数类型
    gtcolor = (0,255,0) ##
    color_1020 = (255,0,0) ##
    color_1110 = (0,0,255) ##
    if is_pred:
        color = (0,0,255)  # Red for predictions
    else:
        color = (0,255,0)  # Green for ground truth
    cv2.drawContours(bevimg, [box], 0, color, 2)
    cv2.arrowedLine(bevimg, center, front_point, color, thickness=None, line_type=None, shift=None, tipLength=None)

    return bevimg

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++ Main Logic and Functions +++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def run_prediction_and_visualization(model, img_list, calib_root, output_dir, args):
    """Original functionality: predict and save visualized images."""
    print("Running prediction and visualization...")
    for img_path in tqdm(img_list, desc="Visualizing"):
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]

        img = cv2.imread(img_path)
        h_img, w_img, _ = img.shape

        ROI = [0, 160, 3840, 1696] if args.roi else [0, 0, w_img, h_img]
        h_roi, w_roi = ROI[3] - ROI[1], ROI[2] - ROI[0]
        h_input, w_input = args.img_size

        results = model.predict(source=img[..., ::-1], save=False, save_txt=False, verbose=False, device=args.device)

        date_name = img_path.split('/')[-3]
        calib_path = os.path.join(calib_root, "20250530", "calib/L2_calib/camera4.json") # date_name
        intrinsic, _, distortion, vehicle_pose = read_calibs(calib_path)
        
        img_2d = img.copy() if args.color_mode == 'bgr' else yuv444_bt601_full_range2rgb(img, w_img, h_img)[..., ::-1].copy()
        img_3d_base = img_2d.copy()
        img_3d_face = img_2d.copy()

        # Create a blank BEV image for each frame
        bevimg = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        ego_center_x, ego_center_y = 500, 1000
        # Dimensions of ego car in pixels (e.g., 4.5m long, 1.8m wide)
        ego_l_px, ego_w_px = 4.5 / 0.1, 1.8 / 0.1
        ego_half_l, ego_half_w = ego_l_px / 2, ego_w_px / 2
        
        ego_box = np.array([
            [ego_center_x - ego_half_w, ego_center_y - ego_half_l],
            [ego_center_x + ego_half_w, ego_center_y - ego_half_l],
            [ego_center_x + ego_half_w, ego_center_y + ego_half_l],
            [ego_center_x - ego_half_w, ego_center_y + ego_half_l]
        ], dtype=np.int32)

        # plot grid by distance, one grid represents 10 meters
        for i in range(11):
            cv2.line(bevimg, (ego_center_x - 500, ego_center_y - i * 100), (ego_center_x + 500, ego_center_y - i * 100), (200, 200, 200), 1)
            cv2.putText(bevimg, f"{i*10}m", (ego_center_x + 5, ego_center_y - i * 100 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # vertical grid, range from -50m to +50m
            cv2.line(bevimg, (ego_center_x - 500 + i * 100, ego_center_y), (ego_center_x - 500 + i * 100, ego_center_y - 1000), (200, 200, 200), 1)
            cv2.putText(bevimg, f"{-50 + i*10}m", (ego_center_x - 500 + i * 100 - 15, ego_center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv2.drawContours(bevimg, [ego_box], 0, (255, 0, 0), -1) # Blue filled rectangle for ego
        # Arrow pointing forward (negative y direction in image space)
        cv2.arrowedLine(bevimg, (ego_center_x, ego_center_y), (ego_center_x, ego_center_y - int(ego_l_px)), (255, 255, 255), 3)

        gt_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                lines = f.readlines()

            gt_list = []
            for line in lines:
                data = line.strip().split(' ')
                if data[0].isdigit():
                    cls_id = int(data[0])
                else:
                    cls_id = args.class_names_str2id[data[0]]  # Changed from function call to dictionary access
                
                # Assuming image size is 3840x2160 for YOLO coord conversion
                # TODO: Make image size dynamic if needed
                w_img, h_img = 3840, 2160
                x_center, y_center, w, h = [float(v) for v in data[1:5]]
                x1 = (x_center - w / 2) * w_img
                y1 = (y_center - h / 2) * h_img
                x2 = (x_center + w / 2) * w_img
                y2 = (y_center + h / 2) * h_img
                box = [x1, y1, x2, y2]
                
                gt3d = np.ones(7) * -1
                if len(data) > 5 and data[5] != '-1':
                    gt3d[:] = [float(v) for v in data[5:12]]

                gt_list.append((cls_id, box, gt3d))

            for gt in gt_list:
                cls_id, box, BB3d = gt
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(img_2d, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                if cls_id <= 16 and args.vis_3d and BB3d[2] > 0:
                    x3d = BB3d[0]
                    y3d = BB3d[1]
                    z3d = BB3d[2]
                    l3d = BB3d[3]
                    h3d = BB3d[4]
                    w3d = BB3d[5]
                    rot_y = BB3d[6]

                    facetype = 'whole'

                    gt3d_cam = np.array([x3d, y3d, z3d, l3d, h3d, w3d, rot_y])
                    # --- BEV Visualization ---
                    # vehicle3d = CamPosToVehiclePos(vehicle_pose['pos'], vehicle_pose['pitch'], vehicle_pose['yaw'], vehicle_pose['roll'], gt3d_cam)
                    # bevimg = drawbev_ego(bevimg, vehicle3d, facetype, is_pred=False)

                    bevimg = drawbev(bevimg, gt3d_cam, facetype, is_pred=False)

        for result in results:
            if result.boxes is None or result.base_decoded is None: continue
            
            boxes = result.boxes
            base3d, faces3d = result.base_decoded, result.faces_decoded
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = np.round(boxes.xyxy[i].cpu().numpy()).astype(int)
                cls_id = int(boxes.cls[i].cpu())

                # if cls_id == 12:
                #     print("Debug: Detected class ID 12")
                
                # --- Visualization ---
                if args.vis_2d:
                    cv2.rectangle(img_2d, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_2d, f'cls:{cls_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

                if cls_id <= 16 and args.vis_3d: # for 3d classes(vehicles, pedestrians, bikes, cyclists)
                    # --- Base 3D Box ---
                    proj_px = int(round(base3d['proj_px'][i][0] * w_roi / w_input)) + ROI[0]
                    proj_pt = (proj_px, int(boxes.xywh[i, 1].cpu()))
                    
                    x3d_norm, y3d_norm = img_cam_kb(np.array([proj_pt[0], proj_pt[1]]).reshape(-1, 2), intrinsic, distortion)
                    z3d = base3d['z3d'][i]
                    infer_x3d, infer_y3d = x3d_norm * z3d, y3d_norm * z3d
                    l3d, h3d, w3d = base3d['l3d'][i], base3d['h3d'][i], base3d['w3d'][i]
                    rot_y = base3d['rot_y'][i]
                    
                    pred3d_cam = np.array([infer_x3d, infer_y3d, z3d, l3d, h3d, w3d, rot_y])
                    corners = cam_corners_front_rear(pred3d_cam, 'whole')
                    pred2d_corners = [cam_img_kb(c[0], c[1], c[2], intrinsic, distortion) for c in corners]
                    
                    sqe = [6, 7, 4, 5, 2, 3, 0, 1]
                    pred2d_corners = np.array(pred2d_corners)[sqe]
                    img_3d_base = drawPointBox(img_3d_base, w_img, h_img, pred2d_corners, colors=[(0, 0, 255), (0, 255, 0)], thickness=1)

                    # --- Face 3D Box ---
                    cutcls = np.argmax(base3d['cutcls_prob'][i])
                    face_scores = faces3d['score_prob'][i]
                    face_idx = np.argmax(face_scores)
                    
                    if cutcls == 1: face_idx = 0
                    elif cutcls == 2: face_idx = 1
                    
                    if cls_id <= 9 and face_scores[face_idx] > 0.2: # only for vehicles with high confidence faces
                        facetypes = ['front', 'tail', 'left', 'right']
                        facetype = facetypes[face_idx]

                        proj_px_face = int(round(faces3d['proj_px'][i][face_idx][0] * w_roi / w_input)) + ROI[0]
                        proj_pt_face = (proj_px_face, int(boxes.xywh[i, 1].cpu()))

                        x3d_norm_f, y3d_norm_f = img_cam_kb(np.array([proj_pt_face[0], proj_pt_face[1]]).reshape(-1, 2), intrinsic, distortion)
                        z3d_face = faces3d['z3d'][i][face_idx]
                        size = faces3d['size'][i][face_idx]
                        
                        if face_idx <= 1: l, h, w = l3d, size[0], size[1]
                        else: l, h, w = size[0], size[1], w3d
                            
                        pred3d_cam_face = np.array([x3d_norm_f * z3d_face, y3d_norm_f * z3d_face, z3d_face, l, h, w, rot_y])
                    else:
                        facetype = 'whole'
                        pred3d_cam_face = pred3d_cam
                        
                    corners_face = cam_corners_front_rear(pred3d_cam_face, facetype)
                    pred2d_corners_face = [cam_img_kb(c[0], c[1], c[2], intrinsic, distortion) for c in corners_face]
                    pred2d_corners_face = np.array(pred2d_corners_face)[sqe]
                    img_3d_face = drawPointBox(img_3d_face, w_img, h_img, pred2d_corners_face, colors=[(0, 0, 255), (0, 255, 0)], thickness=1)

                    # --- BEV Visualization ---
                    # vehicle3d = CamPosToVehiclePos(vehicle_pose['pos'], vehicle_pose['pitch'], vehicle_pose['yaw'], vehicle_pose['roll'], pred3d_cam_face)
                    # bevimg = drawbev_ego(bevimg, vehicle3d, facetype)

                    bevimg = drawbev(bevimg, pred3d_cam_face, facetype)

        if args.vis_2d: cv2.imwrite(os.path.join(output_dir, f"{img_name_no_ext}_vis_2dbox.jpg"), img_2d)
        if args.vis_3d:
            cv2.imwrite(os.path.join(output_dir, f"{img_name_no_ext}_vis_3dbox_base.jpg"), img_3d_base)
            cv2.imwrite(os.path.join(output_dir, f"{img_name_no_ext}_vis_3dbox_face.jpg"), img_3d_face)
            cv2.imwrite(os.path.join(output_dir, f"{img_name_no_ext}_vis_bev.jpg"), bevimg)


def run_evaluation(model, img_list, calib_root, args):
    """New functionality: run prediction and evaluate 3D metrics."""
    print("Running 3D evaluation...")

    label_list = [img_path.replace('/images/', '/labels/').replace('.jpg', '.txt') for img_path in img_list]
    
    gtresult, gtcount = gttransfor3d(label_list, args.class_names_str2id, roi=[0, 160, 3840, 1696])
    dtresult = {}
    num_class = model.model.nc

    for img_path in tqdm(img_list, desc="Inferencing and collecting results"):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h_img, w_img, _ = img.shape
        
        ROI = [0, 160, 3840, 1696] if args.roi else [0, 0, w_img, h_img]
        h_roi, w_roi = ROI[3] - ROI[1], ROI[2] - ROI[0]
        h_input, w_input = args.img_size

        results = model.predict(source=img[..., ::-1], save=False, save_txt=False, verbose=False, device=args.device)
        
        date_name = img_path.split('/')[-3]
        calib_path = os.path.join(calib_root, date_name, "calib/L2_calib/camera4.json")
        intrinsic, _, distortion, _ = read_calibs(calib_path)

        for result in results:
            if result.boxes is None or result.base_decoded is None: continue

            boxes = result.boxes
            base3d, faces3d = result.base_decoded, result.faces_decoded

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].cpu())
                if cls_id not in dtresult: dtresult[cls_id] = []

                x,y,w,h = boxes.xywh[i].cpu().numpy()
                if w < 40 or h < 40: continue

                # --- Decode 3D Face similar to visualization part ---
                # This part is crucial for evaluation
                proj_px = int(round(base3d['proj_px'][i][0] * w_roi / w_input)) + ROI[0]
                proj_pt = (proj_px, int(boxes.xywh[i, 1].cpu()))
                
                cutcls = np.argmax(base3d['cutcls_prob'][i])
                face_scores = faces3d['score_prob'][i]
                face_idx = np.argmax(face_scores)
                
                if cutcls == 1: face_idx = 0
                elif cutcls == 2: face_idx = 1
                
                if cls_id <= 9 and face_scores[face_idx] > 0.2: # vehicle faces only
                    facetypes = ['front', 'tail', 'left', 'right']
                    facetype = facetypes[face_idx]

                    proj_px_face = int(round(faces3d['proj_px'][i][face_idx][0] * w_roi / w_input)) + ROI[0]
                    proj_pt_face = (proj_px_face, int(boxes.xywh[i, 1].cpu()))

                    x3d_norm_f, y3d_norm_f = img_cam_kb(np.array([proj_pt_face[0], proj_pt_face[1]]).reshape(-1, 2), intrinsic, distortion)
                    z3d_face = faces3d['z3d'][i][face_idx]
                    size = faces3d['size'][i][face_idx]

                    if face_idx <= 1: l, h, w = base3d['l3d'][i], size[0], size[1]
                    else: l, h, w = size[0], size[1], base3d['w3d'][i]

                    x, y, z = x3d_norm_f * z3d_face, y3d_norm_f * z3d_face, z3d_face
                elif cls_id <= 16: # Fallback to base prediction
                    facetype = 'whole'
                    x3d_norm, y3d_norm = img_cam_kb(np.array([proj_pt[0], proj_pt[1]]).reshape(-1, 2), intrinsic, distortion)
                    z = base3d['z3d'][i]
                    x, y = x3d_norm * z, y3d_norm * z
                    l, h, w = base3d['l3d'][i], base3d['h3d'][i], base3d['w3d'][i]

                pred3d_parsed = [x, y, z, l, h, w, base3d['rot_y'][i], facetype, cutcls]
                
                dt_item = (
                    img_path,
                    boxes.conf[i].cpu().item(),
                    boxes.xyxy[i].cpu().numpy(),
                    pred3d_parsed,
                    intrinsic, distortion, facetype, args.roi
                )
                dtresult[cls_id].append(dt_item)
    

    # --- Perform evaluation ---
    (precision, recall, aps, conf, lx3d, ldepth, lheading,
     dis_x3d, dis_depth, dis_heading, num_distance) = eval_tp_fp_3drelategap(num_class, gtresult, dtresult, gtcount)

    print("\n" + "="*30)
    print("3D Evaluation Results")
    print("="*30 + "\n")

    mAP = np.mean([ap for ap in aps if ap is not None])
    print(f"mAP @ 0.5: {mAP:.4f}\n")
    for i, ap in enumerate(aps):
        if ap > 0: print(f"Class {i} AP: {ap:.4f}")

    print("\n" + "-"*30 + "\n")

    object_3d_classes = [i for i in range(num_class) if i <= 16 and i in lx3d and lx3d[i]]
    for id in object_3d_classes:
        print(f"--- Class {id} ---")
        print(f"Total X3D Relative Error: {lx3d[id][-1]:.4f}")
        print(f"Total Depth Relative Error: {ldepth[id][-1]:.4f}")
        print(f"Total Heading Average Error (rad): {lheading[id][-1]:.4f}")
        print(f"0-100m X3D Error by distance: {[f'{v:.4f}' for v in dis_x3d[id]]}")
        print(f"0-100m Depth Error by distance: {[f'{v:.4f}' for v in dis_depth[id]]}")
        print(f"0-100m Heading Error by distance: {[f'{v:.4f}' for v in dis_heading[id]]}")
        print(f"Num samples by distance: {num_distance[id]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv11 3D Prediction and Evaluation")
    parser.add_argument('--model-path', type=str, default='./runs/detect_d4q/minieye-driving-d4q-yolo11s-full_image-all-lr0.01-depth50-cls0.82/weights/best.pt', help="Path to the pretrained .pt model file.")
    parser.add_argument('--img-list', type=str, default='./train_data/D4Q/val_D4Q_33_20250530_16465.txt', help="Path to a text file containing a list of image paths.")
    parser.add_argument('--mode', type=str, choices=['eval', 'vis'], default='eval', help="'eval' for 3D metrics, 'vis' for visualization.")
    parser.add_argument('--device', type=str, default='0', help='Device for inference, e.g., "0" for GPU 0, or "cpu".')
    parser.add_argument('--calib-root', type=str, default="/data1/dongying/Mono3d/D4Q_33/", help="Root directory for calibration files.")
    parser.add_argument('--output-dir', type=str, default="./output_val", help="Directory to save visualization results.")
    parser.add_argument('--img-size', type=int, nargs=2, default=[384, 960], help="Input image size for the model [height, width].")
    parser.add_argument('--color-mode', type=str, default='yuv444', choices=['bgr', 'yuv444'], help="Input image color space.")
    parser.add_argument('--roi', default=True, help="Enable ROI cropping.")
    parser.add_argument('--no-vis-2d', dest='vis_2d', action='store_false', help="Disable 2D box visualization.")
    parser.add_argument('--no-vis-3d', dest='vis_3d', action='store_false', help="Disable 3D box visualization.")
    parser.set_defaults(vis_2d=True, vis_3d=True)
    
    args = parser.parse_args()

    class_names = ["car",  "tinycar", "bus", "van", "truck","tanker", "large_truck", "construction_vehicle","special_vehicle", "unknown", # 0-9
                'pedestrian', 'bicycle', "bicyclist", # 10-12
                "motorcycle", "motorcyclist", "tricycle", "tricyclist", # 13-16
                'traffic_light_bbox', 'traffic_light_bulb',
                'traffic_sign', 'animal', 'movable_object',
                'warning_triangle', 'traffic_cone', 'water_barrier', 'crash_barrel',
                'movable_barrier', 'bollard', 'sphere_bollard', 'cube_bollard',
                'cylinder_bollard', 'construction_barrier', 'other_barrier',
                'road_barrier_unknown', "wheel", "plate", "face"
            ]
    # list to dict
    class_names = {i: class_names[i] for i in range(len(class_names))}
    args.class_names_id2str = class_names
    args.class_names_str2id = {v: k for k, v in class_names.items()}

    # Load model
    model = YOLO(args.model_path)
    # onnx_path = model.export(format="onnx")

    if not hasattr(model.model, 'yaml'):
        model.model.yaml = {'channels': 3, 'nc': len(class_names)}

    # Prepare image list
    with open(args.img_list, 'r') as f:
        img_list = [line.strip() for line in f.readlines()]

    # Create output directory
    model_version = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
    output_dir_final = os.path.join(args.output_dir, f"detect_{model_version}")
    os.makedirs(output_dir_final, exist_ok=True)
    
    if args.mode == 'vis':
        run_prediction_and_visualization(model, img_list[:20], args.calib_root, output_dir_final, args)
    elif args.mode == 'eval':
        run_evaluation(model, img_list[:], args.calib_root, args)