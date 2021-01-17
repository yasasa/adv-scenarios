#!/usr/bin/env python
from __future__ import print_function
# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys
try:
    sys.path.append("../carla/dist/carla-0.9.6-py3.7-linux-x86_64.egg")
except IndexError:
    pass
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla
from carla import ColorConverter as cc
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_k
    
    from pygame.locals import K_t   # custom key for NN tesing and manual steeringexcept ImportError:

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
    import time
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

import matplotlib.pyplot as plt


##CNN
from CNN import Net

## Combine RGB and Lidar points
from data_processors import *

import datetime as dt
now = dt.datetime.now()
out_dir = 'random_sampling/%d-%d-%d' % (now.year, now.month, now.day)
# validate_dir = 'validation'
collision_glb = False
########################################################################################
from buffered_saver import BufferedImageSaver

## pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import cv2  # for resizing image

## PID controler as expert
sys.path.append('../carla')

from basic_agent import BasicAgent
from roaming_agent import RoamingAgent

## local planner: turning options, used to chose waypoint (center of lane when turning)
#from local_planner import _retrieve_options
from agents.tools.misc import distance_vehicle, draw_waypoints


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = 'hero' #args.rolename
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.camera_rgb = None
        self.camera_rgb_1 = None
        self.camera_rgb_2 = None
        self.camera_rgb_3 = None
        self.camera_lidar = None
        self.sensor_collision = None
        self.actor_list = []
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.fps = args.fps
        self.restart()
        self.frame = None
        self.delta_seconds = 1.0 / int(args.fps)
        self._queues = []
        self._settings = None
        self.world.wait_for_tick()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            print("spawn point is : " + spawn_point)
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            print('Spawning player at location = ', self.player.get_location())

        self.actor_list.append(self.player)

        print("Initializing custom rgb and lidar sensors")
        bound_y = 0.5 + self.player.bounding_box.extent.y
        sensor_location = carla.Location(x=1.6, z=1.7)

        ## Camera blueprint
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1280')
        bp.set_attribute('image_size_y', '720')

        if self.camera_rgb is not None:
            self.camera_rgb.destroy()
        self.camera_rgb = self.world.spawn_actor(bp, carla.Transform(sensor_location), attach_to=self.player)
        self.actor_list.append(self.camera_rgb)

        if self.camera_rgb_1 is not None:
            self.camera_rgb_1.destroy()
        self.camera_rgb_1 = self.world.spawn_actor(bp, carla.Transform(sensor_location, carla.Rotation(yaw=90)), attach_to=self.player)
        self.actor_list.append(self.camera_rgb_1)

        if self.camera_rgb_2 is not None:
            self.camera_rgb_2.destroy()
        self.camera_rgb_2 = self.world.spawn_actor(bp, carla.Transform(sensor_location, carla.Rotation(yaw=180)), attach_to=self.player)
        self.actor_list.append(self.camera_rgb_2)

        if self.camera_rgb_3 is not None:
            self.camera_rgb_3.destroy()
        self.camera_rgb_3 = self.world.spawn_actor(bp, carla.Transform(sensor_location, carla.Rotation(yaw=270)), attach_to=self.player)
        self.actor_list.append(self.camera_rgb_3)
        
        ## Lidar
        if self.camera_lidar is not None:
            self.camera_lidar.destroy()
        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('range', '5000')
        bp.set_attribute('rotation_frequency', self.fps)
        bp.set_attribute('upper_fov', '10.0')
        bp.set_attribute('lower_fov', '-30.0')
        bp.set_attribute('channels', '32')
        bp.set_attribute('points_per_second', '64000')
        self.camera_lidar = self.world.spawn_actor(bp, carla.Transform(sensor_location), attach_to=self.player)
        self.actor_list.append(self.camera_lidar)

        # ./dagger_data/ep_#/xxx
        # self.rgb_saver = BufferedImageSaver('%s/ep_%d/' % (validate_dir, episode_count),
        #     100, 1280, 720, 3, 'CameraRGB_0')
        # self.rgb_saver_1 = BufferedImageSaver('%s/ep_%d/' % (validate_dir, episode_count),
        #     100, 1280, 720, 3, 'CameraRGB_1')
        # self.rgb_saver_2 = BufferedImageSaver('%s/ep_%d/' % (validate_dir, episode_count),
        #     100, 1280, 720, 3, 'CameraRGB_2')
        # self.rgb_saver_3 = BufferedImageSaver('%s/ep_%d/' % (validate_dir, episode_count),
        #     100, 1280, 720, 3, 'CameraRGB_3')
        # self.lidar_saver = BufferedImageSaver('%s/ep_%d/' % (validate_dir, episode_count), 
        #     100, 8000, 1, 3, 'Lidar')

        print("Initializing collision sensor")
        if self.sensor_collision is not None:
            self.sensor_collision.destroy()
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.sensor_collision = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.player)
        self.sensor_collision.listen(lambda event : self.on_collision(event))
        self.actor_list.append(self.sensor_collision)

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)        
        make_queue(self.camera_rgb.listen)
        make_queue(self.camera_rgb_1.listen)
        make_queue(self.camera_rgb_2.listen)
        make_queue(self.camera_rgb_3.listen)
        make_queue(self.camera_lidar.listen)
        
        return self

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        self.hud.tick(self)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def destroy(self):
        for actor in self.actor_list:
            if actor is not None:
                actor.destroy()

    def on_collision(weak_self, event):
        ## terminate if collide
        global collision_glb
        collision_glb = True

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        self._start_record = False

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]
        if keys[K_q]:
            self._control.reverse = True
        else:
            self._control.reverse = False
        if keys[K_l]:
            self._start_record = True
        if keys[K_k]:
            self._start_record = False


    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 3.333 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, snapshot):
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = snapshot.timestamp.frame_count
        self.simulation_time = snapshot.timestamp.elapsed_seconds

    def tick(self, world):
        self._notifications.tick(self._server_clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            # 'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            # 'Map:     % 20s' % world.world.map_name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            # 'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            # 'Collision:',
            # collision,
            # '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        # lines = __doc__.split('\n')
        # self.font = font
        # self.dim = (680, len(lines) * 22 + 12)
        # self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        # self.seconds_left = 0
        # self.surface = pygame.Surface(self.dim)
        # self.surface.fill((0, 0, 0, 0))
        # for n, line in enumerate(lines):
        #     text_texture = self.font.render(line, True, (255, 255, 255))
        #     self.surface.blit(text_texture, (22, n * 22))
        #     self._render = False
        # self.surface.set_alpha(220)
        pass

    def toggle(self):
        # self._render = not self._render
        pass

    def render(self, display):
        # if self._render:
        #     display.blit(self.surface, self.pos)
        pass

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def clean_lidar(img_lidar):
    tst_inputs = np.frombuffer(img_lidar.raw_data, dtype=np.dtype('f4'))
    tst_inputs = np.reshape(tst_inputs, (int(tst_inputs.shape[0] / 3), 3))
    tst_inputs = np.asarray(tst_inputs, dtype=np.float32)

    # assert if there's large points
    assert(np.max(tst_inputs) < np.sqrt(60.5**2 * 60.5**2))
    # mask super small numbers and make them 0
    # mask = np.absolute(tst_inputs) < 0.0001
    # pts_filter = tst_inputs[mask]
    pts_filter = tst_inputs
    # print(pts_filter.shape)
    if(pts_filter.shape != tst_inputs.shape):
        print('pts filter : pts =', pts_filter.shape, tst_inputs.shape)
        assert(0)
    
    # fig_lidar = plt.figure()
    # ax2 = fig_lidar.add_subplot(111, projection='3d')
    # ax2.scatter(tst_inputs[:, 0], tst_inputs[:, 1], tst_inputs[:, 2], marker='.')
    # ax2.set_xlabel('X Label')
    # ax2.set_ylabel('Y Label')
    # ax2.set_zlabel('Z Label')
    return pts_filter  # n x 3

def clean_img(img_rgb):
    raw_img = np.frombuffer(img_rgb.raw_data, dtype=np.uint8)
    raw_img = raw_img.reshape(720, 1280, 4)
    # get BGR
    raw_img = raw_img[:, :, :3]
    # convert BGR2RGB
    # raw_img = raw_img[:, :, ::-1]
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    # print(raw_img.shape)
    raw_img = np.transpose(raw_img, (1, 0, 2))
    return raw_img   # 1280 x 720 x 3

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    ## Set up pointnet - Lidar
    # num_classes = 1
    # feature_transform = False
    # net = PointFusion(k=1, feature_transform=False)

    # CNN
    net = Net()

    load_pretrained = True

    if load_pretrained:
        weights_path = ('./model/dagger_%d.pth' % args.policy)
        print('loading pretrained model from.. '+ weights_path)
        net.load_state_dict(torch.load(weights_path))
    net.cuda()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        carla_world = client.get_world()

        ## save control signal: throttle, steer, brake, speed

        output_name = "with_cube" if args.cube else "no_cube"

        saver_position = BufferedImageSaver('%s/ep_%d/' % (out_dir, int(args.iter)),
                               100, 1, 1, 3, output_name)

        # saver_control = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
        #                        100, 1, 1, 4, 'Control')

        # validate_lidar_colour = BufferedImageSaver('%s/ep_%d/' % (validate_dir, episode_count),
        #                        100, 4000, 1, 6, 'Coloured_Lidar')

        # validate_all = BufferedImageSaver('%s/ep_%d/' % (validate_dir, episode_count),
        #                        100, 32, 100, 9, 'All')

        # saver_spherical_colour = BufferedImageSaver('%s/ep_%d/' % (out_dir, episode_count),
        #                        100, 32, 100, 4, 'Coloured_Spherical')

        clock = pygame.time.Clock()

        hud = HUD(args.width, args.height)

        world = World(client.get_world(), hud, args)

        controller = KeyboardControl(world, args.autopilot)

        ## PID agent
        # print('current pos =', world.map.get_spawn_points()[0])
        # print('dest pos =', world.map.get_spawn_points()[10])
        
        #world.player.set_transform(world.map.get_spawn_points()[0])
        world.player.set_location(world.map.get_spawn_points()[0].location)
        # print('set: ', world.map.get_spawn_points()[0].location)
        # print('NNPID: current location, ', world.player.get_location())
        
        ## training road
        # world.player.set_transform(carla.Transform(carla.Location(x=155.0, y=129.0, z=2.0), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)))
        # Test random spawn points:
        world.player.set_transform(carla.Transform(carla.Location(x=120.0, y=129.0, z=2.0), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)))
        # Default training road:
        # world.player.set_transform(carla.Transform(carla.Location(x=305.0, y=129.0, z=2.0), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)))
        
        time.sleep(1)

        clock.tick()
        ret = world.tick(timeout=10.0)
        
        #clock.tick(5)

        agent = RoamingAgent(world.player)
        print('NNPID: current location, ', world.player.get_location())

        position = []

        with world:

            while True:
                # clock.tick_busy_loop(60)
                # if controller.parse_events(client, world, clock):
                #     carla_world.tick()
                #     ts = carla_world.wait_for_tick()
                #     return
                if should_quit():
                    return

                ## set custom control
                
                pid_control = agent.run_step()
                waypt_buffer = agent._waypoint_buffer

                while not waypt_buffer:
                    pid_control = agent.run_step()

                global collision_glb
                if collision_glb:
                    player_loc = world.player.get_location()
                    #waypt = carla_world.get_map().get_waypoint(player_loc)
                    waypt, _ = waypt_buffer[0]
                    world.player.set_transform(waypt.transform)
                    #world.player.set_location(waypt.transform.location)
                    collision_glb = False
                    print('hit! respawn')
                    pid_control = agent.run_step()

                pid_control.manual_gear_shift = False
                
                ## Neural net control
                cust_ctrl = controller._control
                cust_ctrl.throttle = 0.6    # 18km/h
                cust_ctrl.brake = 0

                clock.tick()
                ret = world.tick(timeout=10.0)
                if ret:
                    snapshot, img_rgb, img_rgb_1, img_rgb_2, img_rgb_3, img_lidar = ret
                    # snapshot, img_rgb, img_rgb_1, img_rgb_3, img_lidar = ret

                    # save = True

                    points = clean_lidar(img_lidar)
                    images_clean = [clean_img(img_rgb), clean_img(img_rgb_1), clean_img(img_rgb_2), clean_img(img_rgb_3)]
                    # images_clean = [clean_img(img_rgb), clean_img(img_rgb_1), clean_img(img_rgb_3)]

                    # world.lidar_saver.add_image(points, 'Lidar')
                    # world.rgb_saver.add_image(images_clean[0], 'CameraRGB_0')
                    # world.rgb_saver_1.add_image(images_clean[1], 'CameraRGB_1')
                    # world.rgb_saver_2.add_image(images_clean[2], 'CameraRGB_2')
                    # world.rgb_saver_3.add_image(images_clean[3], 'CameraRGB_3')

                    # display the image from the front camera
                    image_surface = pygame.surfarray.make_surface(images_clean[0])
                    display.blit(image_surface, (0, 0))
                    world.hud.render(display)

                    frame_cam1 = snapshot.find(world.camera_rgb_1.id).get_transform().rotation
                    Rot = get_rot_mat(frame_cam1.pitch, -90, frame_cam1.roll)

                    # Colour point cloud
                    curr_input = colour_lidar_quarter(points, Rot, 0, images_clean[0])
                    curr_input = np.concatenate((curr_input, colour_lidar_quarter(points, Rot, 1, images_clean[1])))
                    curr_input = np.concatenate((curr_input, colour_lidar_quarter(points, Rot, 2, images_clean[2])))
                    curr_input = np.concatenate((curr_input, colour_lidar_quarter(points, Rot, 3, images_clean[3])))
                    # validate_lidar_colour.add_image(curr_input, 'Coloured_Lidar')

                    # project lidar point cloud to 2D image
                    curr_input = lidar_project2d(curr_input)  # 3200 x 9
                    # validate_all.add_image(curr_input, 'All')

                    curr_input = curr_input[:,3:7]
                    # saver_spherical_colour.add_image(curr_input, 'Coloured_Spherical')

                    curr_input = np.transpose(curr_input.reshape(32,100,4), (2, 0, 1))
                    tst_inputs = torch.from_numpy(curr_input)

                    # tst_inputs = tst_inputs[0:2400,:]   # 1900 6
                    
                    # tst_inputs = tst_inputs.unsqueeze(0)
                    # tst_inputs = tst_inputs.transpose(2, 1)
                    tst_inputs = tst_inputs.reshape(1, 4, 32, 100)
                    tst_inputs = tst_inputs.cuda()
                    # print(tst_inputs.shape)

                    net = net.eval()
                    outputs = net(tst_inputs.float())
                    outputs = outputs[0].detach().squeeze().tolist()
                    #print(outputs)
                    cust_ctrl.steer = outputs
                    #'''
                    world.player.apply_control(cust_ctrl)

                    player_loc = world.player.get_location()
                    position.append([player_loc.x, player_loc.y, player_loc.z])
                    
                    #'''
                    ## check the center of the lane
                    #waypt = carla_world.get_map().get_waypoint(player_loc)
                    
                    waypt, road_option = waypt_buffer[0]
                    lane_center = waypt.transform.location
                    
                    #print(_current_lane_info.lane_type)
                    #print('waypt ', lane_center)
                    #print('player ', player_loc)
                    
                    dist = math.sqrt((lane_center.x - player_loc.x)**2 + (lane_center.y - player_loc.y)**2)  
                    
                    #print('dist', dist)
                    
                    ## dif in direction
                    next_dir = waypt.transform.rotation.yaw % 360.0
                    player_dir = world.player.get_transform().rotation.yaw % 360.0
                    
                    #print('next_dir, player',next_dir,player_dir)
                    
                    diff_angle = (next_dir - player_dir) % 180.0
                    
                    #print('diff_angle', diff_angle)
                    ## too far from road, use PID control
                    if (diff_angle > 75 and diff_angle < 105) or dist >= 15: 
                        #print('pid_control')
                        #world.player.apply_control(pid_control)
                        #draw_waypoints(carla_world, [waypt], player_loc.z + 2.0)
                        player_loc = world.player.get_location()
                        #waypt = carla_world.get_map().get_waypoint(player_loc)
                        waypt, _ = waypt_buffer[0]
                        world.player.set_transform(waypt.transform)
                        #world.player.set_location(waypt.transform.location)
                        collision_glb = False
                        print('too far! respawn')
                        pid_control = agent.run_step()

                    #''' 
                    # draw_waypoints(carla_world, [waypt], player_loc.z + 2.0)
                    #world.player.apply_control(pid_control)
                else:
                    print("Nothing is returned from world.tick :(")

                ## Record expert (PID) control
                c = pid_control
                throttle = c.throttle  # 0.0, 1.0
                steer = c.steer #-1.0, 1.0
                brake = c.brake # 0.0, 1.0
                
                #print(throttle, steer, brake)
                v = world.player.get_velocity()
                speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                #print('Speed:   % 15.0f km/h' % (speed))
                
                control = np.array([throttle, steer, brake, speed])
                
                saver_position.add_image(np.array(position[-1]), "Position")
                # saver_control.add_image(control, "Control")

                if len(position) == 350:
                    fout = open(out_dir + "/ep_" + str(args.iter) + "/" + output_name + "/end.txt", 'w')
                    fout.write("done")
                    fout.close()
                    break

                pygame.display.flip()

    finally:

        print("Destroying actors...")
        if world is not None:
            world.destroy()

        ## save position
        position = np.asarray(position)
        # save_name = './data/ep_%d/path.npy' % (episode_count)
        # np.save(save_name, position)
        # print('position saved in ',save_name)

        pygame.quit()
        print("Done")


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--policy',
        default=4,
        type=int,
        help='Dagger iteration model (default: 4')
    argparser.add_argument(
        '--iter',
        default=0,
        help='Random sample iteration(default: 0)')
    argparser.add_argument(
        '--fps',
        default='20',
        help='FPS')
    argparser.add_argument(
        '--expnum',
        default=0,
        type=int,
        help='number of experiment run today (default: 0)')
    argparser.add_argument(
        '--cube',
        default=False,
        type=bool,
        help='whether run with cubes spawned (default: False)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    global out_dir
    out_dir += ".%d" % (args.expnum)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
