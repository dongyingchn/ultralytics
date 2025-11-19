import os
import cv2
import numpy as np

import json
import random
from tqdm import tqdm

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

    return intrinsic, extrinsic, distortion

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

def get_image_list():
    random.seed(42)

    data_dir = "/mnt/mono3d/xdzhu_data/Mono3d/Mono3d_4face_2m_g1m3/driving/G1M3_FDL2232/20250912"
    data_dir = "/mnt/mono3d/swji_data/mono3d_test/mono3d/driving"

    vehicle_names = os.listdir(data_dir)
    vehicle_names.sort()

    train_list = []
    val_list = []

    for vehicle_name in vehicle_names:
        if vehicle_name.startswith("."):
            continue
        vehicle_dir = os.path.join(data_dir, vehicle_name)
        if not os.path.isdir(vehicle_dir):
            continue
        
        date_names = os.listdir(vehicle_dir)
        date_names.sort()
        for date_name in date_names:
            if date_name.startswith("."):
                continue
            date_dir = os.path.join(vehicle_dir, date_name)
            if not os.path.isdir(date_dir):
                continue

            image_dir = os.path.join(date_dir, "images")
            img_list = os.listdir(image_dir)
            img_list.sort()
            img_list = [img_name for img_name in img_list if img_name.endswith(".jpg")]
            random.shuffle(img_list)

            num_all = len(img_list)
            num_train = int(num_all * 0.9)
            
            for i in tqdm(range(num_all), desc=f"Processing {vehicle_name}/{date_name}"):
                img_name = img_list[i]
                img_path = os.path.join(image_dir, img_name)

                if i < num_train:
                    train_list.append(img_path)
                else:
                    val_list.append(img_path)

    save_dir = os.path.join("./train_data/D4Q")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_txt_path = os.path.join(save_dir, f"train_20251114_{len(train_list)}.txt")
    with open(train_txt_path, "w") as f:
        for img_path in train_list:
            f.write(f"{img_path}\n")
    print(f"Total training images: {len(train_list)}")

    val_txt_path = os.path.join(save_dir, f"val_20251114_{len(val_list)}.txt")
    with open(val_txt_path, "w") as f:
        for img_path in val_list:
            f.write(f"{img_path}\n")
    print(f"Total validation images: {len(val_list)}")

def get_image_list_2d():
    random.seed(42)

    data_dir = "/mnt/mono3d/swji_data/mono3d_test/Detection"
    
    train_list = []
    val_list = []
    
    image_dir = os.path.join(data_dir, "images")
    img_list = os.listdir(image_dir)
    img_list.sort()
    img_list = [img_name for img_name in img_list if img_name.endswith(".jpg")]
    random.shuffle(img_list)

    num_all = len(img_list)
    num_train = int(num_all * 0.9)
    
    for i in tqdm(range(num_all), desc=f"Processing"):
        img_name = img_list[i]
        img_path = os.path.join(image_dir, img_name)

        if i < num_train:
            train_list.append(img_path)
        else:
            val_list.append(img_path)

    save_dir = os.path.join("./train_data/D4Q")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_txt_path = os.path.join(save_dir, f"train_20251116_{len(train_list)}.txt")
    with open(train_txt_path, "w") as f:
        for img_path in train_list:
            f.write(f"{img_path}\n")
    print(f"Total training images: {len(train_list)}")

    val_txt_path = os.path.join(save_dir, f"val_20251116_{len(val_list)}.txt")
    with open(val_txt_path, "w") as f:
        for img_path in val_list:
            f.write(f"{img_path}\n")
    print(f"Total validation images: {len(val_list)}")

def combine_2d_and_3d():
    train_file_3d = "./train_data/D4Q/train_20251114_195583.txt"
    train_file_2d = "./train_data/D4Q/train_20251116_44640.txt"

    train_file_combined = "./train_data/D4Q/train_20251116_combined.txt"
    with open(train_file_3d, "r") as f:
        lines_3d = f.readlines()
    with open(train_file_2d, "r") as f:
        lines_2d = f.readlines()
    combined_lines = lines_3d + lines_2d

    with open(train_file_combined, "w") as f:
        f.writelines(combined_lines)

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

def check_annos():
    img_path = "/mnt/mono3d/swji_data/mono3d_test/mono3d/driving/D4Q_51/20250712/images/D4Q_51_20250712_seq_262_camera4_001348_139084.jpg"
    img_path = "/mnt/mono3d/swji_data/mono3d_test/mono3d/driving/D4Q_51/20250712/images/D4Q_51_20250712_seq_282_camera4_001029_211596.jpg"
    img_yuv444 = cv2.imread(img_path)
    h_img,w_img = img_yuv444.shape[:2]
    img = yuv444_bt601_full_range2rgb(img_yuv444, w_img, h_img)
    img = img[:,:,::-1].copy()

    label_path = img_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    with open(label_path, "r") as f:
        lines = f.readlines()

    calib_path = "/data1/dongying/Mono3d/D4Q_51/20250712/calib/L2_calib/camera4.json"
    intrinsic, extrinsic, distortion = read_calibs(calib_path)

    # print(lines)
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        line = line.split(" ")
        cls = line[0]

        if cls != "large_truck":
            continue
        # print(line)
        xc_norm, yc_norm = float(line[1]), float(line[2])
        w_norm, h_norm = float(line[3]), float(line[4])

        x1, y1 = int((xc_norm - w_norm / 2) * w_img), int((yc_norm - h_norm / 2) * h_img)
        x2, y2 = int((xc_norm + w_norm / 2) * w_img), int((yc_norm + h_norm / 2) * h_img)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(cls), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(line) < 18:
            continue
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

        face_infos = np.array(line[18:]).reshape(-4, 8)
        face_vis = face_infos[:, 6]
        face_idx = np.argmax(face_vis)

        print(f"face_idx: {face_idx}")

        x3d_face, y3d_face, z3d_face = float(face_infos[face_idx, 0]), float(face_infos[face_idx, 1]), float(face_infos[face_idx, 2])
        xc_face_proj, yc_face_proj = float(face_infos[face_idx, 4]) * w_img, float(face_infos[face_idx, 5]) * h_img

        x3d_new ,y3d_new = img_cam_kb(np.array([xc_face_proj,yc_face_proj]).reshape((-1, 2)),intrinsic, distortion)
        infer_x3d = x3d_new * z3d_face
        infer_y3d = y3d_new * z3d_face

        if face_idx == 0:
            facetype = 'front'
        elif face_idx == 1:
            facetype = 'tail'
        elif face_idx == 2:
            facetype = 'left'
        elif face_idx == 3:
            facetype = 'right'

        # pt_img_with_depth = np.array([[xc_face_proj], [yc_face_proj], [1]]) * float(face_infos[face_idx][2])
        # pt_cam = np.linalg.inv(intrinsic).dot(pt_img_with_depth)
        pred3d_cam_face = np.array([infer_x3d, infer_y3d, float(face_infos[face_idx][2]),
                                l3d, h3d, w3d,
                                rot_y])
        
        # pred3d_cam_face = np.array([x3d_face, y3d_face, z3d_face,
        #                         l3d, h3d, w3d,
        #                         rot_y])

        corners_face = cam_corners_front_rear(pred3d_cam_face, facetype)

        pred2d_corners = []
        for idx_pt in range(len(corners_face)):
            pt_img = cam_img_kb(corners_face[idx_pt,0], corners_face[idx_pt,1], corners_face[idx_pt,2], intrinsic, distortion)
            pred2d_corners.append((pt_img[0], pt_img[1]))

        sqe = [6,7,4,5,2,3,0,1]
        pred2d_corners = np.array(pred2d_corners)[sqe]
        img = drawPointBox(img, w_img, h_img, np.array(pred2d_corners), colors=[(0, 0, 255),(0, 255, 0)], thickness=1)
        cv2.circle(img, (int(xc_face_proj), int(yc_face_proj)), 5, (0, 255, 255), 4)

    img_name = img_path.split("/")[-1]
    cv2.imwrite(img_name.replace(".jpg", "_annos.jpg"), img)

if __name__ == "__main__":
    # get_image_list_2d()
    # combine_2d_and_3d()
    check_annos()