import os
import torch
import json
import queue
from tkinter.messagebox import NO
from typing import List, Union
from pathlib import Path

import cv2
import numpy as np

import carla

from PIL import Image
import imageio
from scipy.spatial.transform import Rotation as R
from cubeadv.sim.sensors.sensor_utils import get_camera_rays, get_ndc_ray_grid, transform_rays_to_camera, yaw_to_mat
from strictfire import StrictFire

    
def setup_folders(base_path):
    dataset_name = base_path
    pose_path = "{}/pose".format(dataset_name)
    depth_path = "{}/depth".format(dataset_name)
    rgb_path = "{}/rgb".format(dataset_name)
    
    if not os.path.exists(pose_path):
        os.makedirs(pose_path, exist_ok=True)
    if not os.path.exists(rgb_path):
        os.makedirs(rgb_path, exist_ok=True)
    if not os.path.exists(depth_path):
        os.makedirs(depth_path, exist_ok=True)
        
    return pose_path, depth_path, rgb_path

def setup_client(host, port):
    cl = carla.Client(host, port)
    world = cl.load_world("Town01")
    return cl, world
    
def setup_weather(world):
    c = world.get_weather()
    c.cloudiness = 0.
    c.precipitation = 0.
    c.precipitation_deposits = 0.
    c.wind_intensity = 0.
    c.sun_azimuth_angle = 250.
    c.fog_density = 0.
    c.sun_altitude_angle = 90.
    c.wetness = 0.
    world.set_weather(c)
    
def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi
    
def globe_sample(count, mid, scale):
    t = np.linspace(-np.pi, np.pi, count)
    x = np.stack([np.cos(t), np.sin(t)], axis=-1)
    
    yaw = wrap_angle(t - np.pi)
    d = np.clip(x * scale[:2], a_min=-scale[:2], a_max=scale[:2])
    p = mid[:2] + d
    points = np.zeros((count, 4))
    points[:, :2] = p
    points[:, 2] = mid[2]
    points[:, 3] = np.rad2deg(yaw)
    
    return points

def get_transforms(nx, ny, nz, nt, midpoint, scale):
    print(nx, ny, nz)
    lo = midpoint - scale / 2
    hi = midpoint + scale / 2
    T0 = np.linspace(lo[0], hi[0], nx)
    T1 = np.linspace(lo[1], hi[1], ny)
    T2 = np.linspace(midpoint[2], hi[2], nz)
    
    yaw = np.linspace(-180, 180, nt, endpoint=False)
    
    poses = np.stack(np.meshgrid(T0, T1, T2, yaw)).reshape(4, -1).transpose()
    return poses
    
def setup_camera(world, H, W, FOV, scale):
    F = W / (2*np.tan(np.deg2rad(FOV/2)))
    metadata = {
        "camera_angle_x": np.deg2rad(FOV),
        "camera_angle_y": np.arctan(H / (2*F)) * 2,
        "fl_x": F,
        "fl_y": F,
        "k1": 0.,
        "k2": 0.,
        "p1": 0.,
        "p2": 0.,
        "cx": W / 2,
        "cy": H / 2,
        "w": W,
        "h": H,
        "scale": 1,
        "white_transparent": False,
        "black_transparent": False,
        "aabb_scale": scale
    }
    
    if world:
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(W))
        camera_bp.set_attribute('image_size_y', str(H))
        camera_bp.set_attribute('fov', str(FOV))
        camera_rgb = world.spawn_actor(camera_bp,
                                       carla.Transform(carla.Location(x=120, y=128, z=2.8), carla.Rotation(yaw=0.)))

        depth_bp = world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(W))
        depth_bp.set_attribute('image_size_y', str(H))
        depth_bp.set_attribute('fov', str(FOV))
        camera_depth = world.spawn_actor(depth_bp,
                                       carla.Transform(carla.Location(x=120, y=128, z=2.8), carla.Rotation(yaw=0.)))

        rgb_queue = queue.Queue()
        depth_queue = queue.Queue()
        camera_rgb.listen(lambda x: rgb_queue.put_nowait(x))
        camera_depth.listen(lambda x: depth_queue.put_nowait(x))
        camera_rgb.queue = rgb_queue
        camera_depth.queue = depth_queue
    else:
        camera_rgb = None
        camera_depth = None
    
    return metadata, camera_rgb, camera_depth
    

def get(frame_id, q):
    item = q.get()
    while item.frame < frame_id:
        item = q.get()
    return item

def capture(world, cam_rgb, cam_depth, tr):
    cam_rgb.set_transform(tr)
    cam_depth.set_transform(tr)
    _ = world.wait_for_tick()
    _ = world.wait_for_tick()
    _ = world.wait_for_tick()
    frame = world.wait_for_tick()
    _ = world.wait_for_tick()
    rgb = get(frame.frame, cam_rgb.queue)
    depth = get(frame.frame, cam_depth.queue)
    return rgb, depth

def convert(data, dim=4, depth=False):
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (data.height, data.width, dim))[:,:,:3]
    
    im = array[:,:,::-1]
    if depth:
        im = im.astype(np.float32)
        im = im[:, :, 0] + im[:, :, 1] * 256 + im[:, :, 2] * 256 * 256
        im = im / (256 * 256 * 256 - 1)
        im = im * 1000
        im = im.astype(np.float32)
    
    return im

def get_matrix(tr, mid, scale):
    carla_tr = np.array(tr.get_matrix())
    
    Rw2c = carla_tr[:3, :3]
    tw  = carla_tr[:3, 3]
    
    tw = (tw - mid) / scale
    
    tw[1] *= -1
    tw[[1, 2]] = tw[[2, 1]]
    
    R_carla_to_ngp = np.array([[0,  -1,   0],
                               [0,   0,   1],
                               [-1.,  0,   0]])
    R_ngp_to_carla = R_carla_to_ngp.T
    
    # Y: Inverting the blender transform that wisp does :/
    blender_matrix = np.array([[1.,  0., 0.],
                               [0.,  0., 1.],
                               [0,  -1., 0.]])
    
    new_tr = np.zeros_like(carla_tr)
    new_tr[3, 3] = 1.
    
    new_tr[:3, :3] = blender_matrix @ Rw2c.T @ R_ngp_to_carla
    new_tr[:3, 3] = tw
    
    return new_tr
    
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm
    
def get_frame(rgb_path, depth_path, mat):
    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)
    frame = {
        "file_path": str(Path(*rgb_path.parts[-2:])),
        "depth_file_path": str(Path(*depth_path.parts[-2:])),
        "transform_matrix": mat.tolist(),
        "sharpness": sharpness(str(rgb_path))
    }
    return frame
    
def save_transforms(path, metadata, frames):
    metadata["frames"] = frames
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
        
def to_carla(p):
    return carla.Transform(carla.Location(x=p[0], y=p[1], z=p[2]), carla.Rotation(yaw=p[3]))
    
def process_depth(depth, tr, scale, midpoint, cam_focal, cam_h, cam_w):
    state = torch.Tensor([tr.location.x, tr.location.y, np.deg2rad(-tr.rotation.yaw)]).view(-1, 3)
    rays = get_ndc_ray_grid(cam_w, cam_h, cam_w / (2*cam_focal))
    rays[:, 2] *= -1
    rays[:, 1] *= -1
   # rays[:, 0] *= -1
    rays_np = rays.numpy().reshape(cam_h, cam_w, 3)
    world_origins, world_rays = transform_rays_to_camera(rays, state, 0.)
    
    depth = depth / rays_np[:, :, 0]
    positions = world_rays * depth.reshape(-1, 1) + world_origins
    positions = positions.numpy()
    
    dist = np.linalg.norm(positions - midpoint, axis=-1)
    dist = dist.reshape(cam_h, cam_w)
    
    depth[dist > scale] = 0.
    depth = depth / scale
    
    return depth #world_rays.numpy().reshape(cam_h, cam_w, 3).astype(np.float32)#positions.reshape(cam_h, cam_w, 3).astype(np.float32)

def process_rgb(rgb, depth_proc):
    rgb[depth_proc < 1e-4] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb
    
def render_and_save(set, transforms, transform_scale, transform_mid,
                    meta, base_path, pose_path, depth_path, rgb_path, 
                    camera_rgb, camera_depth, world, only_poses):
    poses = []
    for i, tr in enumerate(transforms):
        print(i)
        tr = to_carla(tr)
        mat = get_matrix(tr, transform_mid, transform_scale)
        
        rgb_file = f"{rgb_path}/{set}_{i:05d}_raw.png"
        rgb_proc_file = f"{rgb_path}/{set}_{i:05d}.png"
        depth_file = f"{depth_path}/{set}_{i:05d}_raw.exr"
        depth_proc_file = f"{depth_path}/{set}_{i:05d}.exr"
        if not only_poses:
            rgb, depth = capture(world, camera_rgb, camera_depth, tr)
            rgb = convert(rgb) 
            depth = convert(depth, depth=True)
            depth_proc = process_depth(depth.copy(), tr, transform_scale[0], transform_mid, meta['fl_x'], meta['h'], meta['w'])
            rgb_proc = process_rgb(rgb.copy(), depth_proc)
            
            rgbi = Image.fromarray(rgb)
            rgbproci = Image.fromarray(rgb_proc)

            rgbi.save(rgb_file)
            rgbproci.save(rgb_proc_file)
            
            imageio.imwrite(depth_file, depth)
            imageio.imwrite(depth_proc_file, depth_proc)
        poses.append(get_frame(rgb_proc_file, depth_proc_file, mat))

        np.savetxt(f"{pose_path}/{set}_{i:05d}.txt", mat)
        
    save_transforms(f"{base_path}/transform_{set}.json", meta, poses)
    
def generate_images(host:str="localhost", port:int=2000, 
                    dataset_path:str="dataset", height:int=480, width:int=640, 
                    fov:float=90, 
                    prescaling:float=2.,
                    scale:List[List[float]]=[[15, 15, 15]],
                    midpoint:List[List[float]]=[[92.5, 132.5, 2.]],
                    samples: Union[List[List[int]], int]=[[30, 30, 1]],
                    angle_samples:int=4, test_samples:int=20, seed:int=2,
                    only_poses=False):
                    
    np.random.seed(seed)
    world = None
    if not only_poses:
        cl, world = setup_client(host, port)
        setup_weather(world)
        
        
    meta, camera_rgb, camera_depth = setup_camera(world, height, width, fov, 
                                                      prescaling)
                                                      
    pose_path, depth_path, rgb_path = setup_folders(dataset_path)
                                                  
    scale = np.array(scale)
    midpoint = np.array(midpoint)
    if isinstance(samples, list):
        points = [get_transforms(*sample_, angle_samples, mid_, scale_)
                   for sample_, mid_, scale_ in zip(samples, midpoint, scale)]
        points = np.concatenate(points, axis=0)
        lo = midpoint - scale
        hi = midpoint + scale
        lo = np.min(lo, axis=0)
        hi = np.max(hi, axis=0)
        tscale = 0.5*(hi - lo)
        tmid = 0.5*(hi + lo)
    else:
        points = globe_sample(angle_samples, midpoint, scale)
        tscale = scale
        tmid = midpoint
        
    meta["scene_offset"] = tmid.tolist()
    meta["scene_scale"] = tscale.tolist()
 #   meta["scene_scale"][1] *= -1 # To indicate we are flipping the y axis
    
    
    test_mask = np.zeros(points.shape[0])
    test_mask[:test_samples] = 1
    np.random.shuffle(test_mask)
    test_mask = test_mask.astype(bool)
    train_points = points[~test_mask]
    test_points = points[test_mask]
    
    print(f"Capturing {train_points.shape[0]} training transforms")
    render_and_save("train", train_points, tscale, tmid, meta, dataset_path, pose_path, 
                    depth_path, rgb_path, camera_rgb, camera_depth, world, only_poses)
                    
    render_and_save("test", test_points, tscale, tmid, meta, dataset_path, pose_path, 
                    depth_path, rgb_path, camera_rgb, camera_depth, world, only_poses)
                    
    render_and_save("val", test_points, tscale, tmid, meta, dataset_path, pose_path, 
                    depth_path, rgb_path, camera_rgb, camera_depth, world, only_poses)
    
    
if __name__ == '__main__':
    sf = StrictFire(generate_images)
