import os
import cv2
import numpy as np

import json
import random
from tqdm import tqdm

import math

# from predict_3D import read_calibs, cam_corners_front_rear, cam_img_kb, drawPointBox, img_cam_kb

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
    # rot_cam2ego = np.transpose(rot_ego2cam)  # equals np.linalg.inv

    pos = np.array(calib_dict['pos'])
    trans_ego2cam = -1 * np.matmul(rot_ego2cam, pos)

    ego_to_cam = np.zeros((4, 4))
    ego_to_cam[:3, :3] = rot_ego2cam
    ego_to_cam[:3, 3] = trans_ego2cam
    ego_to_cam[3, 3] = 1

    # cam_to_ego = np.linalg.inv(ego_to_cam)

    return ego_to_cam

def read_calibs(calib_path):
    # 判断输入是路径还是字典类型
    if isinstance(calib_path, dict):
        calib_info = calib_path
    elif isinstance(calib_path, str) and os.path.isfile(calib_path):
        with open(calib_path, 'r') as f:
            calib_info = json.load(f)
    else:
        raise ValueError("calib_path should be a file path or a dict")

    intrinsic = np.array([
        [calib_info['focal_u'], 0, calib_info['cu']],
        [0, calib_info['focal_v'], calib_info['cv']],
        [0, 0, 1]
    ], dtype=np.float64)
    extrinsic = get_rot_mat_and_trans(calib_info)
    # rvec_ego2cam, _ = cv2.Rodrigues(extrinsic[:3, :3])
    # tvec_ego2cam = extrinsic[:3, 3]
    distortion = np.array(calib_info['distort_coeffs'])

    vehicle_pose = {
        'pos': calib_info['pos'],
        'pitch': calib_info['pitch'],
        'yaw': calib_info['yaw'],
        'roll': calib_info['roll']
    }

    return intrinsic, extrinsic, distortion, vehicle_pose

def cam_corners_front_rear(pred3d, facetype, camera_pitch):
    dims = pred3d[3:6]
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 1, 0.5]
    if facetype ==  'tail':
        # front
        corners_norm = corners_norm - [0, 0.5, 0.5]
    elif facetype == 'front':
        # tail
        corners_norm = corners_norm - [1, 0.5, 0.5]
    elif facetype == 'right':
        # left
        corners_norm = corners_norm - [0.5, 0.5, 0]
    elif facetype == 'left':
        # right
        corners_norm = corners_norm - [0.5, 0.5, 1]
    elif facetype == 'whole':
        # center
        corners_norm = corners_norm - [0.5, 0.5, 0.5]
    else:
        raise AssertionError('Non valid face type')
    corners = dims.reshape([1, 3]) * corners_norm.reshape([8, 3])
    # rotate around y axis
    corners = rotation_3d_in_axis(corners, pred3d[6], axis=1)
    # corners = rotation_3d_in_axis(corners, camera_pitch * np.pi / 180.0, axis=0)
    corners += pred3d[:3].reshape(1, 3)
    return corners

def img_cam_kb(img_corners,cam_intrinsic,distort_param):   # nx2
    ones_col = np.ones((img_corners.shape[0], 1), dtype=np.float32)
    img_corners = np.concatenate((img_corners.astype(np.float32),ones_col),axis=-1)  # （n,3）
    k1 = distort_param[0]
    k2 = distort_param[1]
    k3 = distort_param[2]
    k4 = distort_param[3]
    points = np.dot(np.linalg.inv(cam_intrinsic),img_corners.T).T          # (2073600, 3)
    xp = points[:,0] # n
    yp = points[:,1] # n
    thetaD = np.sqrt(xp*xp+yp*yp)
    cam_theta,femask = [],[]
    theta = thetaD
    for i in range(20):
        ftheta = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8) - thetaD
        ftheta_dao = 1 + 3 * k1 * theta**2 + 5 * k2 * theta**4 + 7 * k3 * theta**6 + 9 * k4 * theta**8
        theta = theta - ftheta/ftheta_dao
        if abs(ftheta)<0.0000001:
            break
        # print(ftheta)
    
    r = np.tan(theta)
    ###################################
    normx = xp * r / thetaD # 4,H,W
    normy = yp * r / thetaD
    ones_col = np.ones((normy.shape[0], 1), dtype=np.float32)
    x3dy3d = np.concatenate((normx.reshape((-1,1)),normy.reshape((-1,1)),ones_col),axis=-1)
    return x3dy3d[0,0], x3dy3d[0,1]

def cam_img_kb(x, y, z, cam_intrinsic, distort_param):
    # objp=np.array([x/z,y/z,1]).reshape(1,-1,3)
    # rvec = np.array([[[0., 0., 0.]]])
    # tvec = np.array([[[0., 0., 0.]]])
    # image_coord, _ = cv2.fisheye.projectPoints(objp, rvec, tvec,cam_intrinsic, distort_param)
    # image_coord = image_coord.reshape(-1)
    # return image_coord
    camxyz = np.array([x/z, y/z]).reshape(-1,2)
    a = camxyz[...,0]
    b = camxyz[...,1] 
    r = np.sqrt(a * a + b * b)
    theta = np.arctan(r)
    k1 = distort_param[0]
    k2 = distort_param[1]
    k3 = distort_param[2]
    k4 = distort_param[3]
    thetaD = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    xp = thetaD / r * a
    yp = thetaD / r * b
    fx = cam_intrinsic[0,0]
    fy = cam_intrinsic[1,1]
    cx = cam_intrinsic[0,2]
    cy = cam_intrinsic[1,2]
    imgx = (fx * xp + cx)
    imgy = (fy * yp + cy)
    # mask_2d = (imgx>=0)&(imgx<3840)&(imgy>=0)&(imgy<2160)
    pix_point = np.round(np.concatenate((imgx.reshape((-1,1)),imgy.reshape((-1,1))),axis=-1))
    # pix_point = pix_point[mask_2d].astype(np.int32)
    pix_point = pix_point.astype(np.int32)
    return pix_point.reshape(-1)

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

def rotation_3d_in_axis(points, angles, axis=0):
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([
            np.stack([rot_cos, zeros, -rot_sin]),
            np.stack([zeros, ones, zeros]),
            np.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([
            np.stack([rot_cos, -rot_sin, zeros]),
            np.stack([rot_sin, rot_cos, zeros]),
            np.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = np.stack([
            np.stack([zeros, rot_cos, -rot_sin]),
            np.stack([zeros, rot_sin, rot_cos]),
            np.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError(f'axis should in range [0, 1, 2], got {axis}')

    return np.einsum('ij, jk', points, rot_mat_T)

def CamPosToVehiclePos(pos,pitch,yaw,roll,value3d):  ##value3d = [infer_x3d, infer_y3d, z3d, l3d, w3d, h3d, xc, yc, alpha]
    pitch = pitch / 180.0 * np.pi
    yaw = yaw / 180.0 * np.pi
    roll = roll / 180.0 * np.pi

    Rx = np.array([1, 0, 0, 0, np.cos(roll), np.sin(roll), 0, -np.sin(roll), np.cos(roll)]).reshape(3, 3) 
    Ry = np.array([np.cos(pitch), 0, -np.sin(pitch), 0, 1, 0, np.sin(pitch), 0, np.cos(pitch)]).reshape(3, 3)
    Rz = np.array([np.cos(yaw), np.sin(yaw), 0, -np.sin(yaw), np.cos(yaw), 0, 0, 0, 1]).reshape(3, 3)
    vcsgnd2img_mat  = Rx @ Ry @ Rz  #####
    ###########################################xitong###############################################3
    # R2 = np.array([[np.cos(pitch)*np.cos(yaw),np.sin(roll)*np.sin(pitch)*np.cos(yaw)-np.cos(roll)*np.sin(yaw),np.sin(roll)*np.sin(yaw)+ np.cos(roll)*np.sin(pitch)*np.cos(yaw)],
    #       [np.cos(pitch)*np.sin(yaw),np.cos(roll)*np.cos(yaw)+np.sin(roll)*np.sin(pitch)*np.sin(yaw),np.cos(roll)*np.sin(pitch)*np.sin(yaw)-np.sin(roll)*np.cos(yaw)],
    #       [-np.sin(pitch),np.sin(roll)*np.cos(pitch),np.cos(roll)*np.cos(pitch)]])
    # vcsgnd2img_mat = R2.T
    T_mat = np.array(pos)
    r_mat = np.array([0, -1, 0, 0, 0, -1, 1, 0, 0]).reshape(3, 3)
    vcsgnd2img_mat = r_mat @ vcsgnd2img_mat
    tvec_mat = -1 * vcsgnd2img_mat @ T_mat
    vehicle_mat = vcsgnd2img_mat.T @ (np.array([value3d[0], value3d[1], value3d[2]]) - tvec_mat)
    yaw_cam = value3d[-1]
    ##############################
    # yaw_cam_vec = np.array([np.cos(yaw_cam), 0, -np.sin(yaw_cam)]).reshape(3,1)
    # yaw_vehicle_vec = np.dot(vcsgnd2img_mat.T,yaw_cam_vec)
    # vehicle_vec = np.array([1, 0, 0]).reshape(3,1)
    # yaw_cos = 0.0
    # for i in range(3):
    #     yaw_cos += vehicle_vec[i,0] * yaw_vehicle_vec[i,0]
    # yaw_vehicle = np.arccos(yaw_cos)
    # if (np.cos(yaw_cam) > 0):
    #     yaw_vehicle *= -1
    # rotation = (yaw_vehicle) * 180.0 / np.pi
    ############################
    yaw_vehicle = -yaw_cam + yaw - np.pi / 2
    a = math.fmod(yaw_vehicle + np.pi, 2 * np.pi)
    if a < 0:
      a += 2 * np.pi   

    rotation = a - np.pi

    rotation = (rotation) * 180.0 / np.pi

    return [vehicle_mat[0],vehicle_mat[1],vehicle_mat[2],value3d[3],value3d[4],value3d[5],rotation]

def drawbev_ego(bevimg, vehicle3d, face_type, is_pred=True):
    x,y,z,l,h,w = vehicle3d[0], -vehicle3d[1],vehicle3d[2],vehicle3d[3],vehicle3d[4],vehicle3d[5] ###北西天，y取反对应
    yaw = vehicle3d[-1]
    yaw = yaw + 90
    if yaw < 0:
        yaw = yaw + 360
    radians = yaw * (math.pi / 180)

    # 目标图像参数
    W_m = 100  # 目标图像宽度（米）
    H_m = 100  # 目标图像高度（米）
    dx = 0.1  # 每个像素对应的宽度（米）
    dy = 0.1  # 每个像素对应的高度（米）
    if y > 50.0 or y < -50.0 or x > 100.0:
        return bevimg

    transformed_points = [int(500 + y / dx),  1000 - int(x / dy)]

    # # 目标图像内参矩阵
    # K_c = np.array([[1.0 / dx, 0, W_m / (2 * dx)],
    #                 [0, -1.0 / dy, H_m / (2 * dy)],
    #                 [0, 0, 1]], dtype=np.float32)
    
    # R_c = rotation_matrix_from_angles(90,0,0)

    # t_c = np.array([0, 0, 10.0], dtype=np.float32)

    # bevimg = np.zeros((int(H_m / dy), int(W_m / dx), 3), dtype=np.uint8)
    # transformed_points = R_c @ (np.array([x,y,z]).reshape(3, 1)) 
    # transformed_points[2] = 0
    # transformed_points += t_c.reshape(3, 1)
    # transformed_points = K_c @ transformed_points
    # transformed_points = transformed_points[:2, :] / transformed_points[2, :]
    
    H,W = l / dy, w / dx
    if face_type =='whole':
        center = transformed_points
        front_point = [int(center[0] + H * np.cos(radians) / 2.0) , int(center[1] - H * np.sin(radians) / 2.0)]
    elif face_type == 'front':
        center = [int(transformed_points[0] - H * np.cos(radians) / 2.0), int(transformed_points[1] + H * np.sin(radians) / 2.0)]
        front_point = [transformed_points[0], transformed_points[1]]
    elif face_type == 'tail':
        center = [int(transformed_points[0] + H * np.cos(radians) / 2.0), int(transformed_points[1] - H * np.sin(radians) / 2.0)]
        front_point = [int(transformed_points[0] + H * np.cos(radians)), int(transformed_points[1] - H * np.sin(radians))]
    elif face_type == 'left':
        center = [int(transformed_points[0] + W * np.sin(radians) / 2.0), int(transformed_points[1] + W * np.cos(radians) / 2.0)]
        front_point = [int(center[0] + H * np.cos(radians) / 2.0) , int(center[1] - H * np.sin(radians) / 2.0)] 
    elif face_type == 'right':
        center = [int(transformed_points[0] - W * np.sin(radians) / 2.0), int(transformed_points[1] - W * np.cos(radians) / 2.0)]
        front_point = [int(center[0] + H * np.cos(radians) / 2.0) , int(center[1] - H * np.sin(radians) / 2.0)]
    size = [H, W]
    rect = (center, size, yaw)  # 创建旋转矩形的参数结构
    box = cv2.boxPoints(rect)    # 获取四个顶点坐标
    box = np.intp(box)           # 转换为整数类型
    if is_pred:
        color = (0, 0, 255)  # Red for predictions
    else:
        color = (0, 255, 0)  # Green for ground truth
    cv2.drawContours(bevimg, [box], 0, color, 2)

    if is_pred:
        color = (255, 0, 255)  # Red for predictions
    else:
        color = (255, 255, 0)  # Green for ground truth
    cv2.arrowedLine(bevimg, center, front_point, color, thickness=2, line_type=None, shift=None, tipLength=None)

    return bevimg

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
    
def drawbev(bevimg, vehicle3d, face_type, camera_pitch):
    # x,y,z,l,w,h = vehicle3d[0], -vehicle3d[1],vehicle3d[2],vehicle3d[3],vehicle3d[4],vehicle3d[5] ###北西天，y取反对应
    x,y,z,l,h,w = vehicle3d[0], vehicle3d[1],vehicle3d[2],vehicle3d[3],vehicle3d[4],vehicle3d[5] ###右下前，y取反对应
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

    corners = cam_corners_front_rear(np.array([x, y, z, l, h, w, rotation_y]), face_type, camera_pitch)
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
    cv2.drawContours(bevimg, [box], 0, color_1110, 2)
    cv2.arrowedLine(bevimg, center, front_point, color_1110, thickness=None, line_type=None, shift=None, tipLength=None)

    return bevimg

def yuv444_bt601_full_range2rgb(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[1.000, 0.000, 1.402],[1.000, -0.344136, -0.714136],[1.000, 1.772, 0.000]])
    bias = np.array([0, 128, 128])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq-bias_seq),0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

# 'van', 
# '0.10279088541666666', '0.5442231481481481', '0.08166145833333331', '0.07441388888888885', 
# '-18.74231899553714', '0.9015683363835962', '17.486897542674736', 
# '4.964963245945365', '1.899490976837076', '2.055443290313833', 
# '-0.03606498415948515', 
# '0.10104166666666667', '0.5421296296296296', 
# '0.10182291666666667', '0.5782407407407407', 
# '0.7839715027263184', 
# '0', 
# '-16.261451658504924', '0.9015683363835962', '17.576408795843076', '0.7104921947493879', '0.13515625', '0.5439814814814815', '0.6725424739674803', '1.0', 
# '-21.223186332569355', '0.9015683363835962', '17.397386289506397', '0.8480722348565843', '0.071875', '0.5402777777777777', '0.0', '1.0', 
# '-18.779375725993845', '0.9015683363835962', '18.513950890194668', '0.7564502797579481', '0.11380208333333333', '0.5412037037037037', '0.0', '1.0', 
# '-18.705262265080435', '0.9015683363835962', '16.459844195154805', '0.8131001916769355', '0.08776041666666666', '0.5435185185185185', '0.6945245431679085', '1.0'

def get_point_xyz(xyz1,xyz2,num=1):            # num=1为黑色
    (x1,y1,z1),(x2,y2,z2) = xyz1,xyz2
    dis = np.array([abs(x1-x2),abs(y1-y2),abs(z1-z2)])
    dis_idx = dis.argmax()
    res = []
    if dis_idx==0:
        for x in np.concatenate((np.arange(x1,x2,0.01) ,np.arange(x1,x2,-0.01))):
            res.append([x,(x-x1)*(y2-y1)/(x2-x1)+y1,(x-x1)*(z2-z1)/(x2-x1)+z1,num])
    elif dis_idx==1:
        for y in np.concatenate((np.arange(y1,y2,0.01) ,np.arange(y1,y2,-0.01))):
            res.append([(y-y1)*(x2-x1)/(y2-y1)+x1,y,(y-y1)*(z2-z1)/(y2-y1)+z1,num])
    else:
        for z in np.concatenate((np.arange(z1,z2,0.01) ,np.arange(z1,z2,-0.01))):
            res.append([(z-z1)*(x2-x1)/(z2-z1)+x1,(z-z1)*(y2-y1)/(z2-z1)+y1,z,num])
    return res

def xyz_xy(x,y,distort_param,K):
    r = np.sqrt(x * x + y * y)
    r_2 = r * r
    r_4 = r_2 * r_2
    r_6 = r_4 * r_2
    m_distort_coeffs = distort_param
    coeffs = np.zeros(12,dtype=np.float32)
    for i in range(min(m_distort_coeffs.shape[0],12)):
        coeffs[i] = m_distort_coeffs[i]

    radial_ratio = (1 + coeffs[0] * r_2 + coeffs[1] * r_4 + coeffs[4] * r_6) / (1 + coeffs[5] * r_2 + coeffs[6] * r_4 + coeffs[7] * r_6)
    res_x = x * radial_ratio + 2 * coeffs[2] * x * y + coeffs[3] * (r_2 + 2 * pow(x, 2)) + coeffs[8] * r_2 + coeffs[9] * r_4
    res_y = y * radial_ratio + 2 * coeffs[2] * (r_2 + 2 * pow(y, 2)) + coeffs[3] * x * y + coeffs[10] * r_2 + coeffs[11] * r_4
    image_coord = np.dot(K,np.array([res_x,res_y,1])).T
    return image_coord[0:2]

def get_cutoff_corners(cam3d_corners,pts2ds,cam_intrinsic,distortion_param,cam_rectify,imgW,imgH):
    xyz_list = []
    cut_pts2d = np.zeros_like(pts2ds)
    cam3d_corners_update = np.zeros_like(cam3d_corners)
    for (i,j) in [(0, 1), (1, 2), (2, 3), (3, 0),(0,4), (1,5), (2,6),(3,7)]:
        res = get_point_xyz(cam3d_corners[i],cam3d_corners[j],0)
        xyz_list.extend(res)
        if (i,j) in [(0,4), (1,5), (2,6),(3,7)]:
            point = np.array(res)
            pointx_in = np.logical_and(point[:,0]>=-point[:,2]*cam_rectify[0][2]/cam_rectify[0][0],point[:,0]<=point[:,2]*(imgW-1-cam_rectify[0][2])/cam_rectify[0][0])
            pointy_in = np.logical_and(point[:,1]>=-point[:,2]*cam_rectify[1][2]/cam_rectify[1][1],point[:,1]<=point[:,2]*(imgH-1-cam_rectify[1][2])/cam_rectify[1][1])
            pointxy_in = np.logical_and(pointx_in,pointy_in)
            t_point = point[pointxy_in]
            if t_point.shape[0] == 0:
                if len(distortion_param)==5:
                    origimg0 = xyz_xy(point[0,0] /point[0,2], point[0,1] / point[0,2], distortion_param, cam_intrinsic)
                else:
                    origimg0 = cam_img_kb(point[0,0] , point[0,1] , point[0,2],cam_intrinsic,distortion_param)
                cut_pts2d[i] = origimg0[:2]
                cut_pts2d[j] = origimg0[:2]
                continue
            if len(distortion_param)==5:
                point_origimg0 = xyz_xy(t_point[0,0] /t_point[0,2], t_point[0,1] / t_point[0,2], distortion_param, cam_intrinsic)
                point_origimg1 = xyz_xy(t_point[-1,0] /t_point[-1,2], t_point[-1,1] / t_point[-1,2], distortion_param, cam_intrinsic)
            else:
                point_origimg0 = cam_img_kb(t_point[0,0] , t_point[0,1] , t_point[0,2],cam_intrinsic,distortion_param)
                point_origimg1 = cam_img_kb(t_point[-1,0] , t_point[-1,1] , t_point[-1,2],cam_intrinsic,distortion_param)
            cut_pts2d[i] = point_origimg0[:2]
            cut_pts2d[j] = point_origimg1[:2]
            cam3d_corners_update[i] = t_point[0,:3]
            cam3d_corners_update[j] = t_point[-1,:3]
    return cut_pts2d ,cam3d_corners_update

def check_annos():
    txt_path = "/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/train_data/D4Q/val_D4Q_33_20250530_16465.txt"
    save_dir = "./gt_vis/val_D4Q_33_20250530_16465"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(txt_path, "r") as f:
        lines = f.readlines()
        img_paths = [line.strip() for line in lines]

    for img_path in img_paths[:]:

        img_name = os.path.basename(img_path)
        if img_name not in [
            'D4Q_33_20250530_seq_145_camera4_000143_683181.jpg'
        ]:
            continue

        img_yuv444 = cv2.imread(img_path)
        h_img,w_img = img_yuv444.shape[:2]
        img = yuv444_bt601_full_range2rgb(img_yuv444, w_img, h_img)
        img = img[:,:,::-1].copy()
        img_ori = img.copy()

        roi0 = ([0, 160, 3840, 1696], (0,0,255))
        roi1 = ([960, 680, 2880, 1448], (0,255,0))
        roi2 = ([1440, 860, 2400, 1244], (0,255,255))

        for roi, color in [roi0, roi1, roi2]:
            x1, y1, x2, y2 = roi
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), color, 2)
        cv2.imwrite("roi.jpg", img_ori)

        label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        with open(label_path, "r") as f:
            lines = f.readlines()

        calib_path = "/data1/dongying/Mono3d/D4Q_51/20250712/calib/L2_calib/camera4.json"
        # calib_path = "/data1/dongying/Mono3d/D4Q_33/20250530/calib/L2_calib/camera4.json"
        intrinsic, extrinsic, distortion, vehicle_pose = read_calibs(calib_path)
        camera_pitch = vehicle_pose['pitch']

        cam_name = 'camera4'
        if cam_name == 'camera4':
            origH,origW = 2160,3840  
            cam_rectify = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(intrinsic,distortion,[origW,origH],np.eye(3),balance=1) ####
        elif cam_name == 'camera2':
            origH,origW = 1280,1920
            cam_rectify, _ = cv2.getOptimalNewCameraMatrix(intrinsic,distortion, (origW,origH), 1, (origW,origH))
        else:
            print('error cam')

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
        
        cv2.drawContours(bevimg, [ego_box], 0, (255, 0, 0), -1) # Blue filled rectangle for ego
        # Arrow pointing forward (negative y direction in image space)
        cv2.arrowedLine(bevimg, (ego_center_x, ego_center_y), (ego_center_x, ego_center_y - int(ego_l_px)), (255, 255, 255), 3)

        # print(lines)
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            line = line.split(" ")
            cls = line[0]

            # if cls != "tricycle":
            #     continue
            # print(line)
            xc_norm, yc_norm = float(line[1]), float(line[2])
            w_norm, h_norm = float(line[3]), float(line[4])

            x1, y1 = int((xc_norm - w_norm / 2) * w_img), int((yc_norm - h_norm / 2) * h_img)
            x2, y2 = int((xc_norm + w_norm / 2) * w_img), int((yc_norm + h_norm / 2) * h_img)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(cls), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if len(line) < 18:
                continue

            if len(line) >= 18:
                
                x3d, y3d, z3d = float(line[5]), float(line[6]), float(line[7])
                l3d, h3d, w3d = float(line[8]), float(line[9]), float(line[10])
                rot_y = float(line[11])
                xc_proj, yc_proj = float(line[12]), float(line[13])

                delta_0 = rot_y
                delta_1 = rot_y - np.pi / 2
                delta_2 = rot_y + np.pi / 2
                ang_mask = int(abs(rot_y - np.pi) < abs(rot_y + np.pi))
                delta_3 = (rot_y - np.pi)*ang_mask + (rot_y + np.pi)*(1-ang_mask)
                angles = np.array([delta_0, delta_1, delta_2, delta_3])
                # angle classification
                ang_cls = (np.pi*0.5 - abs(angles)) / (np.pi*0.5)
                ang_cls[ang_cls < 0] = 0

                print(cls, (x1,y1,x2,y2), rot_y, ang_cls, angles)

                gt3d_cam = np.array([x3d, y3d, z3d, l3d, h3d, w3d, rot_y])
                facetype = 'whole'

            # elif len(line) > 18:
            #     face_infos = np.array(line[18:]).reshape(-4, 8)
            #     face_vis = face_infos[:, 6]
            #     face_idx = np.argmax(face_vis)

            #     print(f"face_idx: {face_idx}")

            #     x3d_face, y3d_face, z3d_face = float(face_infos[face_idx, 0]), float(face_infos[face_idx, 1]), float(face_infos[face_idx, 2])
            #     xc_face_proj, yc_face_proj = float(face_infos[face_idx, 4]) * w_img, float(face_infos[face_idx, 5]) * h_img

            #     x3d_new ,y3d_new = img_cam_kb(np.array([xc_face_proj,yc_face_proj]).reshape((-1, 2)),intrinsic, distortion)
            #     infer_x3d = x3d_new * z3d_face
            #     infer_y3d = y3d_new * z3d_face

            #     if face_idx == 0:
            #         facetype = 'front'
            #     elif face_idx == 1:
            #         facetype = 'tail'
            #     elif face_idx == 2:
            #         facetype = 'left'
            #     elif face_idx == 3:
            #         facetype = 'right'

            #     l3d, h3d, w3d = float(line[8]), float(line[9]), float(line[10])
            #     rot_y = float(line[11])

            #     # pt_img_with_depth = np.array([[xc_face_proj], [yc_face_proj], [1]]) * float(face_infos[face_idx][2])
            #     # pt_cam = np.linalg.inv(intrinsic).dot(pt_img_with_depth)
            #     gt3d_cam = np.array([infer_x3d, infer_y3d, float(face_infos[face_idx][2]),
            #                         l3d, h3d, w3d,
            #                         rot_y])
            
            # pred3d_cam_face = np.array([x3d_face, y3d_face, z3d_face,
            #                         l3d, h3d, w3d,
            #                         rot_y])

            corners = cam_corners_front_rear(gt3d_cam, facetype, camera_pitch)

            pred2d_corners = []
            for idx_pt in range(len(corners)):
                pt_img = cam_img_kb(corners[idx_pt,0], corners[idx_pt,1], corners[idx_pt,2], intrinsic, distortion)
                pred2d_corners.append((pt_img[0], pt_img[1]))

            #####deal cutoff####
            x_left = corners[:,0]>=-corners[:,2]*cam_rectify[0][2]/cam_rectify[0][0]                       # (8,)
            x_right = corners[:,0]<=corners[:,2]*(origW-1-cam_rectify[0][2])/cam_rectify[0][0]
            have_in = ((x_left & x_right)[[0,1,4,5]]==True).sum()
            # po2ds = None
            if have_in >= 1 and have_in < 4:
                cut_pts2d,cam3d_corners_update= get_cutoff_corners(corners,pred2d_corners,intrinsic,distortion,cam_rectify,origW,origH)
                if cut_pts2d is not None:
                    pred2d_corners = cut_pts2d
            
            sqe = [6,7,4,5,2,3,0,1]
            pred2d_corners = np.array(pred2d_corners)[sqe]
            img = drawPointBox(img, w_img, h_img, np.array(pred2d_corners), colors=[(0, 0, 255),(0, 255, 0)], thickness=1)
            # cv2.circle(img, (int(xc_face_proj), int(yc_face_proj)), 5, (0, 255, 255), 4)

            # vehicle3d = CamPosToVehiclePos(vehicle_pose['pos'], vehicle_pose['pitch'], vehicle_pose['yaw'], vehicle_pose['roll'], gt3d_cam)
            # bevimg = drawbev(bevimg, vehicle3d, facetype, is_pred=False)

            bevimg = drawbev(bevimg, gt3d_cam, facetype, camera_pitch)

        img_name = img_path.split("/")[-1]

        cv2.imwrite(os.path.join(save_dir, img_name.replace(".jpg", "_annos.jpg")), img)
        cv2.imwrite(os.path.join(save_dir, img_name.replace(".jpg", "_bev.jpg")), bevimg)

if __name__ == "__main__":
    # get_image_list()
    # get_image_list_2d()
    # combine_2d_and_3d()
    check_annos()