import os
import numpy as np
from scipy.spatial import Delaunay


camera_ls = [2, 3]

_sem2label = {
    'Misc': -1,
    'Car': 0,
    'Van': 0,
    'Truck': 2,
    'Tram': 3,
    'Pedestrian': 4
}

def kitti_string_to_float(str):
    return float(str.split('e')[0]) * 10**int(str.split('e')[1])


def get_rotation(roll, pitch, heading):
    s_heading = np.sin(heading)
    c_heading = np.cos(heading)
    rot_z = np.array([[c_heading, -s_heading, 0], [s_heading, c_heading, 0], [0, 0, 1]])

    s_pitch = np.sin(pitch)
    c_pitch = np.cos(pitch)
    rot_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])

    s_roll = np.sin(roll)
    c_roll = np.cos(roll)
    rot_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])

    rot = np.matmul(rot_z, np.matmul(rot_y, rot_x))

    return rot


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0., 0., 0., 1.]])])


def get_camera_poses_tracking(poses_velo_w_tracking, tracking_calibration, scene_no=None, exp=False):
    camera_poses = []

    opengl2kitti = np.array([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

    #####################
    # Debug Camera offset
    if scene_no == 2:
        yaw = np.deg2rad(0.7) ## Affects camera rig roll: High --> counterclockwise
        pitch = np.deg2rad(-0.5) ## Affects camera rig yaw: High --> Turn Right
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(0.9) ## Affects camera rig pitch: High -->  up
        # roll = np.deg2rad(1.2)
    elif scene_no == 1:
        if exp:
            yaw = np.deg2rad(0.3) ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.6) ## Affects camera rig yaw: High --> Turn Right
            # pitch = np.deg2rad(-0.97)
            roll = np.deg2rad(0.75) ## Affects camera rig pitch: High -->  up
            # roll = np.deg2rad(1.2)
        else:
            yaw = np.deg2rad(0.5)  ## Affects camera rig roll: High --> counterclockwise
            pitch = np.deg2rad(-0.5)  ## Affects camera rig yaw: High --> Turn Right
            roll = np.deg2rad(0.75)  ## Affects camera rig pitch: High -->  up
    else:
        yaw = np.deg2rad(0.05)
        pitch = np.deg2rad(-0.75)
        # pitch = np.deg2rad(-0.97)
        roll = np.deg2rad(1.05)
        #roll = np.deg2rad(1.2)

    cam_debug = np.eye(4)
    cam_debug[:3, :3] = get_rotation(roll, pitch, yaw)

    Tr_cam2camrect = tracking_calibration['Tr_cam2camrect']
    Tr_cam2camrect = np.matmul(Tr_cam2camrect, cam_debug)
    Tr_camrect2cam = np.linalg.inv(Tr_cam2camrect)
    Tr_velo2cam = tracking_calibration['Tr_velo2cam']
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)

    camera_poses_imu = []
    for cam in camera_ls:
        Tr_camrect2cam_i = tracking_calibration['Tr_camrect2cam0' + str(cam)]
        Tr_cam_i2camrect = np.linalg.inv(Tr_camrect2cam_i)
        # transform camera axis from kitti to opengl for nerf:
        # cam_i_camrect = np.matmul(Tr_cam_i2camrect, opengl2kitti)
        cam_i_cam0 = np.matmul(Tr_camrect2cam, Tr_cam_i2camrect)
        cam_i_velo = np.matmul(Tr_cam2velo, cam_i_cam0)

        cam_i_w = np.matmul(poses_velo_w_tracking, cam_i_velo)
        camera_poses_imu.append(cam_i_w)

    return np.concatenate(camera_poses_imu, axis=0)

def calib_from_txt(calibration_path):

    c2c = []

    f = open(os.path.join(calibration_path, 'calib_cam_to_cam.txt'), "r")
    cam_to_cam_str = f.read()
    [left_cam, right_cam] = cam_to_cam_str.split('S_02: ')[1].split('S_03: ')
    cam_to_cam_ls = [left_cam, right_cam]

    for i, cam_str in enumerate(cam_to_cam_ls):
        r_str, t_str = cam_str.split('R_0' + str(i + 2) + ': ')[1].split('\nT_0' + str(i + 2) + ': ')
        t_str = t_str.split('\n')[0]
        R = np.array([kitti_string_to_float(r) for r in r_str.split(' ')])
        R = np.reshape(R, [3, 3])
        t = np.array([kitti_string_to_float(t) for t in t_str.split(' ')])
        Tr = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0., 0., 0., 1.])[None, :]])


        t_str_rect, s_rect_part = cam_str.split('\nT_0' + str(i + 2) + ': ')[1].split('\nS_rect_0' + str(i + 2) + ': ')
        s_rect_str, r_rect_part = s_rect_part.split('\nR_rect_0' + str(i + 2) + ': ')
        r_rect_str = r_rect_part.split('\nP_rect_0' + str(i + 2) + ': ')[0]
        R_rect = np.array([kitti_string_to_float(r) for r in r_rect_str.split(' ')])
        R_rect = np.reshape(R_rect, [3, 3])
        t_rect = np.array([kitti_string_to_float(t) for t in t_str_rect.split(' ')])
        Tr_rect = np.concatenate([np.concatenate([R_rect, t_rect[:, None]], axis=1), np.array([0., 0., 0., 1.])[None, :]])


        c2c.append(Tr_rect)

    c2leftRGB = c2c[0]
    c2rightRGB = c2c[1]

    f = open(os.path.join(calibration_path, 'calib_velo_to_cam.txt'), "r")
    velo_to_cam_str = f.read()
    r_str, t_str = velo_to_cam_str.split('R: ')[1].split('\nT: ')
    t_str = t_str.split('\n')[0]
    R = np.array([kitti_string_to_float(r) for r in r_str.split(' ')])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(' ')])
    v2c = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0., 0., 0., 1.])[None, :]])

    f = open(os.path.join(calibration_path, 'calib_imu_to_velo.txt'), "r")
    imu_to_velo_str = f.read()
    r_str, t_str = imu_to_velo_str.split('R: ')[1].split('\nT: ')
    R = np.array([kitti_string_to_float(r) for r in r_str.split(' ')])
    R = np.reshape(R, [3, 3])
    t = np.array([kitti_string_to_float(r) for r in t_str.split(' ')])
    imu2v = np.concatenate([np.concatenate([R, t[:, None]], axis=1), np.array([0., 0., 0., 1.])[None, :]])

    focal = kitti_string_to_float(left_cam.split('P_rect_02: ')[1].split()[0])

    return imu2v, v2c, c2leftRGB, c2rightRGB, focal


def tracking_calib_from_txt(calibration_path):
    f = open(calibration_path)
    calib_str = f.read().splitlines()
    calibs = []
    for calibration in calib_str:
        calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

    P0 = np.reshape(calibs[0], [3, 4])
    P1 = np.reshape(calibs[1], [3, 4])
    P2 = np.reshape(calibs[2], [3, 4])
    P3 = np.reshape(calibs[3], [3, 4])

    Tr_cam2camrect = np.eye(4)
    R_rect = np.reshape(calibs[4], [3, 3])
    Tr_cam2camrect[:3, :3] = R_rect
    Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0., 0., 0., 1.]])], axis=0)
    Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0., 0., 0., 1.]])], axis=0)

    return {'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3, 'Tr_cam2camrect': Tr_cam2camrect,
            'Tr_velo2cam': Tr_velo2cam, 'Tr_imu2velo': Tr_imu2velo}


def get_poses_calibration(basedir, oxts_path_tracking=None, selected_frames=None):

    def oxts_to_pose(oxts):
        poses = []

        def latlon_to_mercator(lat, lon, s):
            r = 6378137.0
            x = s * r * ((np.pi * lon) / 180)
            y = s * r * np.log(np.tan((np.pi * (90 + lat)) / 360))
            return [x, y]

        if selected_frames == None:
            lat0 = oxts[0][0]
            scale = np.cos(lat0 * np.pi / 180)
            pose_0_inv = None
        else:
            oxts0 = oxts[selected_frames[0][0]]
            lat0 = oxts0[0]
            scale = np.cos(lat0 * np.pi / 180)

            pose_i = np.eye(4)

            [x, y] = latlon_to_mercator(oxts0[0], oxts0[1], scale)
            z = oxts0[2]
            translation = np.array([x, y, z])
            rotation = get_rotation(oxts0[3], oxts0[4], oxts0[5])
            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])

        for oxts_val in oxts:
            pose_i = np.zeros([4, 4])
            pose_i[3, 3] = 1

            [x, y] = latlon_to_mercator(oxts_val[0], oxts_val[1], scale)
            z = oxts_val[2]
            translation = np.array([x, y, z])

            roll = oxts_val[3]
            pitch = oxts_val[4]
            heading = oxts_val[5]
            rotation = get_rotation(roll, pitch, heading)

            pose_i[:3, :] = np.concatenate([rotation, translation[:, None]], axis=1)
            if pose_0_inv is None:
                pose_0_inv = invert_transformation(pose_i[:3, :3], pose_i[:3, 3])
                # pose_0_inv = np.linalg.inv(pose_i)

            pose_i = np.matmul(pose_0_inv, pose_i)
            poses.append(pose_i)

        return np.array(poses)

    if oxts_path_tracking is None:
        oxts_path = os.path.join(basedir, 'oxts/data')
        oxts = np.array([np.loadtxt(os.path.join(oxts_path, file)) for file in sorted(os.listdir(oxts_path))])
        calibration_path = os.path.dirname(basedir)

        calibrations = calib_from_txt(calibration_path)

        focal = calibrations[4]

        poses = oxts_to_pose(oxts)

    ### Tracking oxts
    else:
        oxts_tracking = np.loadtxt(oxts_path_tracking)
        poses = oxts_to_pose(oxts_tracking)
        calibrations = None
        focal = None
        # Set velodyne close to z = 0
        # poses[:, 2, 3] -= 0.8

    return poses, calibrations, focal


def get_obj_pose_tracking(tracklet_path, poses_imu_tracking, calibrations):

    def roty_matrix(roty):
        c = np.cos(roty)
        s = np.sin(roty)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    velo2cam = calibrations['Tr_velo2cam']
    imu2velo = calibrations['Tr_imu2velo']
    cam2velo = invert_transformation(velo2cam[:3, :3], velo2cam[:3, 3])
    velo2imu = invert_transformation(imu2velo[:3, :3], imu2velo[:3, 3])

    objects_meta_kitti = {}
    objects_meta = {}
    tracklets_ls = []

    f = open(tracklet_path)
    tracklets_str = f.read().splitlines()

    n_scene_frames = len(poses_imu_tracking)
    n_obj_in_frame = np.zeros(n_scene_frames)

    # Metadata for all objects in the scene
    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        if float(tracklet[1]) < 0:
            continue
        id = tracklet[1]
        if tracklet[2] in _sem2label:
            type = _sem2label[tracklet[2]]
            if not int(id) in objects_meta_kitti:
                height = tracklet[10]
                width = tracklet[11]
                length = tracklet[12]
                objects_meta_kitti[int(id)] = np.array([float(id), type, length, height, width])

            tr_array = np.concatenate([np.array(tracklet[:2]).astype(np.float), np.array([type]), np.array(tracklet[3:]).astype(np.float)])
            tracklets_ls.append(tr_array)
            n_obj_in_frame[int(tracklet[0])] += 1

    # Objects in each frame
    tracklets_array = np.array(tracklets_ls)

    max_obj_per_frame = int(n_obj_in_frame.max())
    visible_objects = np.ones([(n_scene_frames) * 2, max_obj_per_frame, 18]) * -1.
    visible_objects_box2world = np.ones([n_scene_frames, max_obj_per_frame, 4, 4]) * -1.


    for tracklet in tracklets_array:
        frame_no = tracklet[0]
        obj_id = tracklet[1]
        frame_id = np.array([frame_no])
        id_int = int(obj_id)
        obj_type = np.array([objects_meta_kitti[id_int][1]])
        dim = objects_meta_kitti[id_int][-3:].astype(np.float32) 

        if id_int not in objects_meta:
            objects_meta[id_int] = np.concatenate([np.array([id_int]).astype(np.float32),
                                                    objects_meta_kitti[id_int][2:].astype(np.float64),
                                                    np.array([objects_meta_kitti[id_int][1]]).astype(np.float64)])

        pose = tracklet[-4:] # location (3 dimension), rotation_y(1 dimension) in camera coordinate

        obj_pose_c = np.eye(4)
        obj_pose_c[:3, 3] = pose[:3] # location
        roty = pose[3]
        obj_pose_c[:3, :3] = roty_matrix(roty) # box2cam

        obj_pose_imu = np.matmul(velo2imu, np.matmul(cam2velo, obj_pose_c)) # box2imu

        pose_imu_w_frame_i = poses_imu_tracking[int(frame_id)]

        pose_obj_w_i = np.matmul(pose_imu_w_frame_i, obj_pose_imu) # box2world

        yaw_aprox = -np.arctan2(pose_obj_w_i[1, 0], pose_obj_w_i[0, 0])

        # TODO: Change if necessary
        is_moving = 1.

        pose_3d = np.array([pose_obj_w_i[0, 3],
                            pose_obj_w_i[1, 3],
                            pose_obj_w_i[2, 3],
                            yaw_aprox, 0, 0, is_moving])
        pose_3d_cam = pose

        
        for j, cam in enumerate(camera_ls):
            cam = np.array(cam).astype(np.float32)[None]
            obj = np.concatenate([frame_id, cam, np.array([obj_id]), obj_type, dim, pose_3d, pose])
            frame_cam_id = int(frame_no) + j * n_scene_frames
            obj_column = np.argwhere(visible_objects[frame_cam_id, :, 0] < 0).min()
            visible_objects[frame_cam_id, obj_column] = obj
            
            if j == 0:
                visible_objects_box2world[int(frame_no), obj_column] = pose_obj_w_i



        

    # Remove not moving objects
    print('Removing non moving objects')
    obj_to_del = []
    for key, values in objects_meta.items():
        all_obj_poses = np.where(visible_objects[:, :, 2] == key)
        if len(all_obj_poses[0]) > 0 and values[4] != 4.:
            frame_intervall = all_obj_poses[0][[0, -1]]
            y = all_obj_poses[1][[0, -1]]
            obj_poses = visible_objects[[frame_intervall, y]][:, 7:10]
            distance = np.linalg.norm(obj_poses[1] - obj_poses[0])
            print(distance)
            if distance < 0.5:
                print('Removed:', key)
                obj_to_del.append(key)
                visible_objects[all_obj_poses] = np.ones(18) * -1.

    for key in obj_to_del:
        del objects_meta[key]

    return visible_objects, objects_meta, visible_objects_box2world 

def get_scene_images(basedir, seq):
    imgs = []

    left_img_path = f'image_02/{seq}'
    right_img_path = f'image_03/{seq}'

    for frame_no in range(len(os.listdir(os.path.join(basedir, left_img_path)))):
            for pth in [left_img_path, right_img_path]:
                frame_dir = os.path.join(basedir, pth)
                frame = sorted(os.listdir(frame_dir))[frame_no]
                fname = os.path.join(frame_dir, frame)
                imgs.append(fname)

    return imgs

def bilinear_interpolate_numpy(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T


def bbox_rect_to_lidar(bbox, R0, V2C):
    '''
    Borrowed from https://github1s.com/junming259/Partial_Point_Clouds_Generation/blob/main/calibration.py#L69-L84
    
    Transform the bbox from camera system to the lidar system.

    Args:
        bbox: [N, 7], [x, y, z, dx, dy, dz, heading]

    Returns:
        bbox_lidar: [N, 7]
    '''
    def rect_to_lidar(pts_rect, R0, V2C):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = cart_to_hom(pts_rect)  # (N, 4)

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0, V2C).T))
        return pts_lidar[:, 0:3] / pts_lidar[:, 3:4] 

    loc, dims, rots = bbox[:, :3], bbox[:, 3:6], bbox[:, 6]
    loc_lidar = rect_to_lidar(loc, R0, V2C)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2 # converting box location to box center
    bbox_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    return bbox_lidar

def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        corners3d: (N, 8, 3)
    """
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ]) / 2

    corners3d = boxes3d[:, None, 3:6] * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d, boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3)
        angle: (B), angle along z-axis, angle increases x ==> y

    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    ones = np.ones_like(angle, dtype=np.float32)
    zeros = np.zeros_like(angle, dtype=np.float32)
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points, rot_matrix)
    return points_rot

def is_within_3d_box(points, corners3d):
    """
    Check whether a point is within bbox

    Args:
        points:  (N, 3)
        corners3d: (M, 8, 3), corners of M bboxes.

    Returns:
        flag: (M, N), bool
    """
    num_objects = len(corners3d)
    flag = []
    for i in range(num_objects):
        hull = Delaunay(corners3d[i])
        flag.append(hull.find_simplex(points) >= 0)
    try:
        flag = np.stack(flag, axis=0)
    except:
        return None
    return flag

def points_to_canonical(points, box):
    '''
    Transform points within bbox to a canonical pose and normalized scale

    Args:
        points: (N, 3), points within bbox
        box: (7), box parameters

    Returns:
        points_canonical: (N, 3)
    '''
    center = box[:3].reshape(1, 3)
    rot = -box[-1].reshape(1)
    points_centered = (points - center).reshape(1, -1, 3)
    points_centered_rot = rotate_points_along_z(points_centered, rot)
    scale = (1 / np.abs(box[3:6]).max()) * 0.999999
    points_canonical = points_centered_rot * scale

    box_canonical = box.copy()
    box_canonical[:3] = 0
    box_canonical[-1] = 0
    box_canonical = box_canonical * scale
    return points_canonical.squeeze(), box_canonical
