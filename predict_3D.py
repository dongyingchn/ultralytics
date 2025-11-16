import cv2
from ultralytics import YOLO

import os
import json
import numpy as np
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

    return intrinsic, extrinsic, distortion

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

def cam_corners_front_rear(pred3d, facetype):
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
    corners += pred3d[:3].reshape(1, 3)
    return corners

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

def yuv444_bt601_full_range2rgb(src, img_w, img_h):
    src_seq = np.transpose(np.reshape(src, (-1,3)), (1,0)).astype(np.float32)
    trans_mat = np.array([[1.000, 0.000, 1.402],[1.000, -0.344136, -0.714136],[1.000, 1.772, 0.000]])
    bias = np.array([0, 128, 128])
    bias_seq = np.reshape(np.repeat(bias, img_h*img_w), (3, img_h*img_w))
    dst_seq = np.minimum(np.maximum(np.matmul(trans_mat, src_seq-bias_seq),0),255)
    dst_seq = np.transpose(dst_seq, (1, 0))
    dst = np.round(np.reshape(dst_seq, (img_h, img_w, 3))).astype(np.uint8)
    return dst

# Load a model
pretrained_path = "/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect/train21/weights/best.pt"
pretrained_path = "/deeplearning_team/ydong/dongying/projects/monocular_3d_object_detection/ultralytics/runs/detect_d4q/minieye-driving-d4q-roi-2d2/weights/best.pt"
model_version = pretrained_path.split('/')[-3]
model = YOLO(pretrained_path)  # load a pretrained model, YOLO11n model

# Run batched inference on a list of images

img_path = "/mnt/mono3d/xdzhu_data/Mono3d/Mono3d_4face_2m_g1m3/driving/G1M3_FDL2232/20250912/images/G1M3_FDL2232_20250912_seq_53_camera4_002777_68656.jpg"
img_path = "/mnt/mono3d/swji_data/mono3d_test/mono3d/driving/D4Q_51/20250712/images/D4Q_51_20250712_seq_95_camera4_000632_80721.jpg"
img = cv2.imread(img_path)
# img = cv2.resize(img, (960, 540))
h_img, w_img, _ = img.shape

ROI = [0, 160, 3840, 1696]  # x1,y1,x2,y2
if ROI is not None:
    h_roi, w_roi = ROI[3] - ROI[1], ROI[2] - ROI[0]
else:
    ROI = [0, 0, w_img, h_img]
    h_roi, w_roi = img.shape[:2]

results = model.predict(source=img, save=True, save_txt=True)  # return a list of Results objects

calib_path = "/data1/dongying/Mono3d/G1M3_FDL2232/20250912/calib/L2_calib/camera4.json"
calib_path = "/data1/dongying/Mono3d/D4Q_51/20250712/calib/L2_calib/camera4.json"
intrinsic, extrinsic, distortion = read_calibs(calib_path)

mode = 'yuv444'
if mode == 'bgr':
    img_2d = img.copy()
elif mode == 'yuv444':
    img_2d = yuv444_bt601_full_range2rgb(img.copy(), w_img, h_img)[..., ::-1].copy()
img_3d_base = img_2d.copy()
img_3d_face = img_2d.copy()

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

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    
    base3d = result.base_decoded
    faces3d = result.faces_decoded

    xywh = boxes.xywh.detach().cpu().numpy()  # xywh numpy array
    xyxy = boxes.xyxy.detach().cpu().numpy()  # xyxy numpy array
    cls = boxes.cls.detach().cpu().numpy().astype(int)  # class indices numpy array
    for i in range(len(boxes)):

        if cls[i] > 16:
            continue

        x, y, w, h = np.round(xywh[i]).astype(int)
        x1,y1,x2,y2 = np.round(xyxy[i]).astype(int)
        
        proj_px = int(round(base3d['proj_offset_cell'][i][0] * w_roi))
        proj_pt = (proj_px+ROI[0], y)

        cv2.rectangle(img_2d, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_2d, 'cls:'+str(cls[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
        # cv2.circle(img, proj_pt, 5, (0, 0, 255), 2)

        pt_img_with_depth = np.array([[proj_pt[0]], [proj_pt[1]], [1]]) * base3d['z3d'][i]
        pt_cam = np.linalg.inv(intrinsic).dot(pt_img_with_depth)

        l3d, h3d, w3d = base3d['l3d'][i], base3d['h3d'][i], base3d['w3d'][i]
        rot_y = base3d['rot_y'][i]

        pred3d_cam = np.array([pt_cam[0][0], pt_cam[1][0], base3d['z3d'][i],
                            l3d, h3d, w3d,
                            rot_y])
        
        corners = cam_corners_front_rear(pred3d_cam, 'whole')

        pred2d_corners = []
        for idx_pt in range(len(corners)):
            pt_img = cam_img_kb(corners[idx_pt,0], corners[idx_pt,1], corners[idx_pt,2], intrinsic, distortion)
            pred2d_corners.append((pt_img[0], pt_img[1]))

        sqe = [6,7,4,5,2,3,0,1]
        pred2d_corners = np.array(pred2d_corners)[sqe]
        img_3d_base = drawPointBox(img_3d_base, w_img, h_img, np.array(pred2d_corners), colors=[(0, 0, 255),(0, 255, 0)], thickness=1)

        cutcls = np.argmax(base3d['cutcls_prob'][i])
        
        # decode face
        face_scores = faces3d['score_prob'][i]
        face_idx = np.argmax(face_scores)

        if cutcls == 1:
            face_idx = 0  # front face
        elif cutcls == 2:
            face_idx = 1  # tail face

        if face_scores[face_idx] > 0.2:

            proj_px = int(round(faces3d['proj_offset_cell'][i][face_idx][0] * w_roi))
            proj_pt = (proj_px+ROI[0], y)

            pt_img_with_depth = np.array([[proj_pt[0]], [proj_pt[1]], [1]]) * faces3d['z3d'][i][face_idx]
            pt_cam = np.linalg.inv(intrinsic).dot(pt_img_with_depth)

            size = faces3d['size'][i][face_idx]

            if face_idx == 0:
                facetype = 'front'
                l, h, w = l3d, size[0], size[1]
            elif face_idx == 1:
                facetype = 'tail'
                l, h, w = l3d, size[0], size[1]
            elif face_idx == 2:
                facetype = 'left'
                l, h, w = size[0], size[1], w3d
            elif face_idx == 3:
                facetype = 'right'
                l, h, w = size[0], size[1], w3d

            pred3d_cam_face = np.array([pt_cam[0][0], pt_cam[1][0], faces3d['z3d'][i][face_idx],
                                        l, h, w,
                                        rot_y])
        else:
            facetype = 'whole'
            pred3d_cam_face = pred3d_cam

        corners_face = cam_corners_front_rear(pred3d_cam_face, facetype)

        pred2d_corners = []
        for idx_pt in range(len(corners_face)):
            pt_img = cam_img_kb(corners_face[idx_pt,0], corners_face[idx_pt,1], corners_face[idx_pt,2], intrinsic, distortion)
            pred2d_corners.append((pt_img[0], pt_img[1]))

        sqe = [6,7,4,5,2,3,0,1]
        pred2d_corners = np.array(pred2d_corners)[sqe]
        img_3d_face = drawPointBox(img_3d_face, w_img, h_img, np.array(pred2d_corners), colors=[(0, 0, 255),(0, 255, 0)], thickness=1)
    
    cv2.imwrite(f"result_{model_version}_vis_2dbox.jpg", img_2d)
    cv2.imwrite(f"result_{model_version}_vis_3dbox_base.jpg", img_3d_base)
    cv2.imwrite(f"result_{model_version}_vis_3dbox_face.jpg", img_3d_face)
    
    # result.show()  # display to screen
    # result.save(filename="result_train10.jpg")  # save to disk