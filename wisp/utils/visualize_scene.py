import torch
import os
import numpy as np
import skimage
import imageio
#from wisp.ops.image import write_png, write_exr
import matplotlib.pyplot as plt

from wisp.config_parser import get_modules_from_config, get_optimizer_from_config

def parse_args():
    from wisp.config_parser import parse_options, argparse_to_str

    # Usual boilerplate
    parser = parse_options(return_parser=True)

    args, args_str = argparse_to_str(parser)
    return args, args_str

# from https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
def convert_from_uvd(u, v, d):
    focalx = 380.0
    focaly = 380.0
    cx = 320.0
    cy = 180.0

    #d *= self.pxToMetre
    x_over_z = (cx - u) / focalx
    y_over_z = (cy - v) / focaly
    #z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    z = d
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z

    # z = d
    # inv_camera_mat = np.array([[1/fx, 0, -cx*fy/fx/fy, 0],
    #                            [0, 1/fy, -cy/fy, 0],
    #                            [0, 0, 1, 0]
    #                            [0, 0, 0, 1]], dtype=np.float)
    # xyz1 = z * inv_camera_mat * np.array([u,v,1,1/z])

if __name__ == '__main__':
    args, args_str = parse_args()
    pipeline, train_dataset, device = get_modules_from_config(args)

    rays = train_dataset.data["rays"]
    ray_os = list(rays.origins)
    ray_ds = list(rays.dirs)
    img_shape = train_dataset.img_shape
    print('img shape:', img_shape)

    xs = []
    ys = []
    zs = []

    for i, key in enumerate(train_dataset.data["cameras"].keys()):
        ray_o = ray_os[i]#.reshape(*img_shape[:2], 3)
        ray_d = ray_ds[i]#.reshape(*img_shape[:2], 3)
        #print(ray_o.shape)
        #print(ray_d.shape)
        depth_path = os.path.join(args.dataset_path, 'depth', "{}.png".format(key))
        print(depth_path)
        
        img = skimage.io.imread(depth_path, as_gray=True)
        #print(img.shape)
        #im_show = (img * 255).astype(np.uint8)
        #imageio.imwrite('tmp_depth_out.png',im_show)

        for u in range(img_shape[0]):
            for v in range(img_shape[1]):
                d = img[u,v] * 255
                x,y,z = convert_from_uvd(u, v, d)
                #print(x,y,z)
                xs.append(x)
                ys.append(y)
                zs.append(z)

        # ray_o = ray_o.numpy()
        # ray_d = ray_d.numpy()
        # img = img.reshape(-1, 1)
        
        # xyz = ray_o + img * ray_d
        # print('xyz shape', xyz.shape)

        #break

    #xs = np.array(xs)
    #ys = np.array(ys)
    #zs = np.array(zs)

    # img = img.reshape(-1)
    # indices = np.where(img < 0.01)[0]
    # xyz = xyz[indices, :]
    # print('xyz filtered shape', xyz.shape)
    # xs = xyz[:,0]
    # ys = xyz[:,1]
    # zs = xyz[:,2]

    print('plotting')

    f = plt.figure(figsize=(22,6))
    # ax1 = plt.subplot(1, 4, 1, projection='3d')
    # #ax = f.add_subplot(projection='3d')
    # ax1.scatter(xs, ys, zs)
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Z')

    ax2 = plt.subplot(1, 4, 2)
    ax2.scatter(xs,ys)

    ax3 = plt.subplot(1,4,3)
    ax3.scatter(xs,zs)

    ax4 = plt.subplot(1,4,4)
    ax4.scatter(ys,zs)

    f.savefig("bov_scene.png", bbox_inches='tight')

    



