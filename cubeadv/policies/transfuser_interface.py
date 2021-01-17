from re import A
from cubeadv.policies.transfuser.team_code_transfuser.model import LidarCenterNet
from cubeadv.sim.sensors import Lidar, Camera, SensorRig
from .transfuser.team_code_transfuser.latentTF import latentTFBackbone
from .transfuser.team_code_transfuser.config import GlobalConfig

import torch
from torchvision.transforms import Resize, CenterCrop


from .transfuser.team_code_transfuser.data import lidar_to_histogram_features, draw_target_point, lidar_bev_cam_correspondences

# possibly the worst way to draw a circle
def draw_circle(image, center, radius, color, thickness):
    batch_size = image.shape[0]
    height, width = image.shape[-2], image.shape[-1]
    px, py = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='xy')
    #px = width / 2
    #py = height / 2
    index_grid = torch.stack([px, py], dim=-1).type_as(image)
    offset = (index_grid - center.float().view(batch_size, 1, 1, -1)).norm(dim=-1)
    
    mask = radius - thickness / 2 < offset 
    mask *= offset < radius + thickness / 2
    image[mask] = color
    
    return image

def draw_target_point(target_point, color = 255):
    image = torch.zeros(target_point.shape[0], 256, 256, dtype=torch.uint8).type_as(target_point)
    target_point = target_point

    # convert to lidar coordinate
    target_point[:, 1] += 1.3
    point = target_point * 8.
    point[:, 1] *= -1
    point[:, 1] = 128 - point[:, 1] 
    point[:, 0] += 128 
    point = point.int()
    point = point.clamp(0, 256)
    
    draw_circle(image, point, radius=5, color=color, thickness=3)
    image = image.reshape(-1, 256, 256)
    return image

class Transfuser:
    def __init__(self, goal=None):
        self.config = GlobalConfig(setting='eval')
        
        if(self.config.sync_batch_norm == True):
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) # Model was trained with Sync. Batch Norm. Need to convert it otherwise parameters will load incorrectly.
        self.config.n_layer = 4
        self.config.use_target_point_image = True
        net = LidarCenterNet(self.config, 'cuda', 'latentTF', 'regnety_032', 'regnety_032', False)
        state_dict = torch.load("/home/yasasa/models_2022/latentTF/model_seed1_41.pth", map_location='cuda:0')
        state_dict = {k[7:]: v for k, v in state_dict.items()} # Removes the .module coming from the Distributed Training. Remove this if you want to evaluate a model trained without DDP.
        net.load_state_dict(state_dict, strict=False)
        net.cuda()
        self.net = net
        self.goal = goal
    
    def forward(self, state, image):
        wps = self.run_step(state, image, self.goal.unsqueeze(0))
        wps = self.to_world(state, wps)
        return wps
    
    def shift_x_scale_crop(self, image, scale, crop, crop_shift=0):
        crop_h, crop_w = crop
        batch, _height, _width, channels = image.shape
        (width, height) = (int(_width // scale), int(_height // scale))
        image = image.permute(0, 3, 1, 2)
        if scale != 1:
            image = Resize((height, width))(image)
        start_y = height//2 - crop_h//2
        start_x = width//2 - crop_w//2
        
        # only shift in x direction
        #start_x += int(crop_shift // scale)
        cropped_image = image[:, :, start_y:start_y+crop_h, start_x:start_x+crop_w]
        return cropped_image
    
    def prepare_image(self, image):
        image_degrees = []
        
        crop_shift = 0
        rgb = self.shift_x_scale_crop(
                            image, scale=self.config.scale, 
                            crop=self.config.img_resolution, 
                            crop_shift=crop_shift)
            
        return rgb
        
    def to_bev(self, state, x):
        theta = state[:, 2] + torch.pi/2
        
        Rs = torch.zeros(state.shape[0], 2, 2).type_as(state)
        Rs[:, 0, 0] = theta.cos()
        Rs[:, 0, 1] = -theta.sin()
        Rs[:, 1, 0] = theta.sin()
        Rs[:, 1, 1] = theta.cos()
        
        local_x = (x - state[:, :2]).unsqueeze(1)
        local_x = (local_x @ Rs).squeeze(1)
        return local_x
        
    def to_world(self, state, p):
        theta = state[:, 2] + torch.pi/2
        Rs = torch.zeros(state.shape[0], 2, 2).type_as(state)
        Rs[:, 0, 0] = theta.cos()
        Rs[:, 0, 1] = -theta.sin()
        Rs[:, 1, 0] = theta.sin()
        Rs[:, 1, 1] = theta.cos()
        local_x = p
        local_x = (local_x @ Rs.mT).squeeze()
        
        state_ret = torch.zeros(state.shape[0], p.shape[1], state.shape[-1]).type_as(state)
        state_ret[:, :, :2] = local_x + state[:, :2].unsqueeze(1)
        
        local_x[:, :2]
        angles = state_ret[:, :-1, :2]
        state_ret[:, :, 2] =  state[:, None, 2]
        return state_ret
        
        
    def prepare_goal_location(self, target_point):
        current_target_point = target_point
        target_point_image = draw_target_point(current_target_point)

        return target_point_image, target_point
        
    def scale_crop(self, image, scale=1, start_x=0, crop_x=None, start_y=0, crop_y=None):
        batch, cameras, _height, _width, channels = image.shape
        (width, height) = (_width // scale, _height // scale)
        if scale != 1:
            image.permute(0, 1, 4, 2, 3) # N, Cam, Chann, H, W
            image = Resize((width, height))(image.view(-1, channels, _height, _width))
            image = image.view(batch, cameras, channels, height, width)
            image = image.permute(0, 1, 3, 4, 2) # permute back
        if crop_x is None:
            crop_x = width
        if crop_y is None:
            crop_y = height
            
        start_y = height//2 - crop_y//2
        start_x = width//2 - crop_x//2
            
        cropped_image = image[:, :, start_y:start_y+crop_y, start_x:start_x+crop_x]
        return cropped_image
        
    def run_step(self, state, images, goal):
        batch_size = state.shape[0]
        local_goal = self.to_bev(state, goal)
        # prepare image input
        image = self.scale_crop(images, 
                                self.config.scale, crop_x = self.config.img_width, 
                                crop_y = self.config.img_width // 2)
        
                                
        images = image.permute(0, 2, 1, 3, 4).reshape(image.shape[0], image.shape[2], -1, 3)
        
        image = self.prepare_image(images)

        num_points = None
        lidar_bev = torch.zeros(batch_size, 2, self.config.lidar_resolution_width, self.config.lidar_resolution_height).type_as(state)

        # prepare goal location input
        target_point_image, target_point = self.prepare_goal_location(local_goal)
        target_point_image = target_point_image.view(-1, 1, 256, 256)
        

        # prepare velocity input
        gt_velocity = torch.Tensor([1.]).type_as(state)
        velocity = gt_velocity.view(1, 1).expand(state.shape[0], -1) # used by transfuser

        # forward pass
        with torch.no_grad():
            rotated_bb = []
            pre_pred_wp, rotated_bb = self.net.forward_ego(image, lidar_bev, target_point, target_point_image, velocity, num_points=num_points)
            pred_wp = self.to_world(state, pre_pred_wp)
        
        
        return pre_pred_wp, pred_wp, (target_point_image, image, target_point)
        
    def __call__(self, image, state):
        return self.run_step(state, image, self.goal)[1]