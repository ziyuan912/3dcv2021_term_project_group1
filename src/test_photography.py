import open3d as o3d
import numpy as np
import json
import os
from scipy.spatial.transform import Rotation
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.path import Path
from tqdm import tqdm

base_folder = './testcases/2'
img_folder = os.path.join(base_folder, 'images')
json_file = os.path.join(base_folder, 'sfm_data.json')
bg_folder = './3d_photography/demo'
save_folder = os.path.join(base_folder, 'final_result')
save_params = False
draw_camera = True
draw_path = False

os.makedirs(save_folder, exist_ok=True)

R_euler = np.array([0, 0, 0]).astype(float)
t = np.array([0, 0, 0]).astype(float)
scale_x, scale_y, scale_z = 1.0, 1.0, 1.0

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# add cube
cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
cube_vertices = np.asarray(cube.vertices).copy()
vis.add_geometry(cube)

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
    return axes

def get_transform_mat(rotation, translation, scale_x, scale_y, scale_z):
    r_mat = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3)
    scale_mat[0, 0], scale_mat[1, 1], scale_mat[2, 2] = scale_x, scale_y, scale_z
    transform_mat = np.concatenate([r_mat @ scale_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

def update_cube():
    global cube, cube_vertices, R_euler, t, scale_x, scale_y, scale_z

    transform_mat = get_transform_mat(R_euler, t, scale_x, scale_y, scale_z)
    
    transform_vertices = (transform_mat @ np.concatenate([
                            cube_vertices.transpose(), 
                            np.ones([1, cube_vertices.shape[0]])
                            ], axis=0)).transpose()

    cube.vertices = o3d.utility.Vector3dVector(transform_vertices)
    cube.compute_vertex_normals()
    cube.paint_uniform_color([1, 0.706, 0])
    vis.update_geometry(cube)

def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1: # key down
        shift_pressed = True
    elif action == 0: # key up
        shift_pressed = False
    return True

def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cube()

def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cube()

def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cube()

def update_scale_x(vis):
    global scale_x, shift_pressed
    scale_x += -0.05 if shift_pressed else 0.05
    update_cube()

def update_scale_y(vis):
    global scale_y, shift_pressed
    scale_y += -0.05 if shift_pressed else 0.05
    update_cube()

def update_scale_z(vis):
    global scale_z, shift_pressed
    scale_z += -0.05 if shift_pressed else 0.05
    update_cube()

# set key callback
shift_pressed = False
vis.register_key_action_callback(340, toggle_key_shift)
vis.register_key_action_callback(344, toggle_key_shift)
vis.register_key_callback(ord('A'), update_tx)
vis.register_key_callback(ord('S'), update_ty)
vis.register_key_callback(ord('D'), update_tz)
vis.register_key_callback(ord('Z'), update_rx)
vis.register_key_callback(ord('X'), update_ry)
vis.register_key_callback(ord('C'), update_rz)
vis.register_key_callback(ord('V'), update_scale_x)
vis.register_key_callback(ord('B'), update_scale_y)
vis.register_key_callback(ord('N'), update_scale_z)

axes = load_axes()
vis.add_geometry(axes)

def main():
    global t, R_euler, scale_x, scale_y, scale_z

    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    intrinsics = json_data['intrinsics']
    cam_info = intrinsics[0]['value']['ptr_wrapper']['data']
    img_size = [cam_info['height'], cam_info['width']]
    K = np.zeros((3, 3), dtype=float)
    K[0, 2] = cam_info['principal_point'][0]
    K[1, 2] = cam_info['principal_point'][1]
    K[0, 0] = K[1, 1] = cam_info['focal_length']
    K[2, 2] = 1
    
    extrinsics = json_data['extrinsics']
    Rs, Ts = [], []
    for extrinsic in extrinsics:
        R = np.array(extrinsic["value"]["rotation"], dtype=float).reshape((3, 3))
        T = np.array(extrinsic["value"]["center"], dtype=float).reshape((1, 3))
        Rs.append(R)
        Ts.append(T)
    Rs = np.array(Rs)
    Ts = np.array(Ts)

    if os.path.isfile(os.path.join(base_folder, 'params.npy')):
        with open(os.path.join(base_folder, 'params.npy'), 'rb') as f:
            params = np.load(f)
        t = params[0]
        R_euler = params[1]
        scale_x, scale_y, scale_z = params[2]
        update_cube()
    
    max_y, max_x, min_y, min_x = 0, 0, img_size[0], img_size[1]
    plane_center_index = [0, 1, 5, 4]
    cam_path = []
    for idx, (R, T) in enumerate(zip(Rs, Ts)):
        if draw_camera and idx % 10 == 0: # camera pyramid
            points = np.array([[0, 0, 0], [0, img_size[0], 1], [img_size[1], img_size[0], 1], [img_size[1], 0, 1], [0, 0, 1]])
            points = (np.linalg.inv(K) @ points.T)
            points = (np.linalg.inv(R) @ (points + R@T.T)).T
            lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
            colors = np.array([[1, 0, 0] for i in range(len(lines))])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)
        
        origin = np.array([[0, 0, 0]])
        origin = (np.linalg.inv(K) @ origin.T)
        origin = (np.linalg.inv(R) @ (origin + R@T.T)).T
        cam_path.append(origin)

        P = np.concatenate([R, (-R @ T.T)], axis=1)
        cube_points = np.asarray(cube.vertices)
        cube_points = np.hstack([cube_points, np.ones((cube_points.shape[0], 1), dtype=np.float32)]).astype(np.float32)
        project_points = (K @ P @ cube_points.T).T
        project_points = np.divide(project_points, project_points[:, -1, np.newaxis]).astype(int)[:, :2]
        for i, point in enumerate(project_points):
            if i in plane_center_index:
                max_x = min(max(max_x, point[0]), img_size[1])
                max_y = min(max(max_y, point[1]), img_size[0])
                min_x = max(min(min_x, point[0]), 0)
                min_y = max(min(min_y, point[1]), 0)

    # draw camera path
    if draw_path:
        lines = []
        for i in range(len(cam_path)-1):
            lines.append([i, i+1])
        lines = np.array(lines)
        colors = np.array([[0, 1, 0] for i in range(len(lines))])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(cam_path)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    img2_size = [max_x-min_x, max_y-min_y]

    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    vc.convert_from_pinhole_camera_parameters(vc_cam)

    points = []
    colors = []
    imgs = []
    for v in json_data['views']:
        img_path = v["value"]["ptr_wrapper"]["data"]['filename']
        img = cv2.imread(os.path.join('./test2/frame3', img_path))
        imgs.append(img)

    print('read image finish.')

    for p in json_data['structure']:
        points.append(p['value']['X'])
        key = int(p["value"]["observations"][0]["key"])
        uv = p["value"]["observations"][0]["value"]["x"]
        img = imgs[key]
        color = img[int(uv[1]), int(uv[0]), :][::-1]
        color = color / 255
        colors.append(color)

    points = np.array(points, dtype=float)
    colors = np.array(colors, dtype=float)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

    params = np.zeros([3, 3])
    params[0] = t
    params[1] = R_euler
    params[2] = [scale_x, scale_y, scale_z]

    if not os.path.isfile(os.path.join(base_folder, 'params.npy')) or save_params:
        np.save(os.path.join(base_folder, 'params.npy'), params)

    img_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpeg')])

    for idx, (img_file, R, T) in enumerate(tqdm(zip(img_files, Rs, Ts), total=len(Rs))):
        P = np.concatenate([R, (-R @ T.T)], axis=1)

        img = cv2.imread(os.path.join(img_folder, img_file))
        cube_points = np.asarray(cube.vertices)
        cube_points = np.hstack([cube_points, np.ones((cube_points.shape[0], 1), dtype=np.float32)]).astype(np.float32)
        project_points = (K @ P @ cube_points.T).T
        project_points = np.divide(project_points, project_points[:, -1, np.newaxis]).astype(int)[:, :2]
        for i, point in enumerate(project_points):
            if i in plane_center_index:
                cv2.circle(img, (point[0], point[1]), 10, (0, 0, 255), -1)
            # cv2.putText(img, str(i), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 1, cv2.LINE_AA)

        line_connect = [[0, 1], [4, 5], [0, 4], [1, 5]]
        # line_connect = [[0, 1], [0, 2], [2, 3], [1, 3], [4, 5], [6, 7], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
        for line in line_connect:
            cv2.line(img, (project_points[line[0]][0], project_points[line[0]][1]), (project_points[line[1]][0], project_points[line[1]][1]), (0, 0, 255), 3)
        
        img2 = cv2.imread(os.path.join(bg_folder, "%05d"%idx+'.png'))
        scale_x = img2_size[0] / img2.shape[1]
        scale_y = img2_size[1] / img2.shape[0]

        if scale_x > scale_y:
            img2 = cv2.resize(img2, (int(img2.shape[1]*scale_x), int(img2.shape[0]*scale_x)))
        else:
            img2 = cv2.resize(img2, (int(img2.shape[1]*scale_y), int(img2.shape[0]*scale_y)))

        bg = np.zeros_like(img)
        bg[min_y:max_y, min_x:max_x, :] = img2[0:img2_size[1], 0:img2_size[0], :]
        img2 = bg

        poly = [(point[0], point[1]) for point in project_points[plane_center_index]]
        x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0])) # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T 

        p = Path(poly, closed=False) # make a polygon
        grid = p.contains_points(points)
        for i, v in enumerate(grid):
            if v == True:
                img[i//img.shape[1], i%img.shape[1], :] = img2[i//img.shape[1], i%img.shape[1], :] 
        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
        
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_folder, "%05d"%idx + '.png'), img)

if __name__ == '__main__':
    main()
    