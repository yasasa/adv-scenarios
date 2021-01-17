#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import subprocess
import sys
try:
    sys.path.append("../carla/dist/carla-0.9.6-py3.7-linux-x86_64.egg")
except IndexError:
    pass

import carla
import numpy as np
import argparse
import logging
import random
from generate_spawn_points import get_building_spawn_points, get_spawn_points_from_file

import datetime as dt
now = dt.datetime.now()
out_dir = 'random_sampling/%d-%d-%d' % (now.year, now.month, now.day)

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
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
        '-b', '--number-of-buildings',
        metavar='B',
        default=2,
        type=int,
        help='number of buildings (default: 5)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--filterb',
        metavar='PATTERN',
        default='building.mybuilding.cube',
        help='building filter (default: "building.mybuilding.*")')
    argparser.add_argument(
        '--iter',
        default=0,
        type=int,
        help='Random sample iteration(default: 0)')
    argparser.add_argument(
        '--expnum',
        default=0,
        type=int,
        help='number of experiment run today (default: 0)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    global out_dir
    out_dir += ".%d" % (args.expnum)

    building_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    try:
        world = client.get_world()
        blueprintsBuildings = world.get_blueprint_library().filter(args.filterb)
        building_bp = blueprintsBuildings[0]

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn buildings
        # --------------
        # Get the spawn points for cubes: location + rotation
        spawn_points = []
        test_run = False
        if test_run:
            building_spawn_points = get_building_spawn_points()  # 3000 x 3
            number_of_spawn_points = building_spawn_points.shape[0]

            for i in range(args.number_of_buildings):
                loc = building_spawn_points[i*50]
                # spawn_point = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]), carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0))
                if i != 0:
                    spawn_point = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]))
                else:
                    spawn_point = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]), carla.Rotation(pitch=45))
                spawn_points.append(spawn_point)
        else:
            filename = "random_sampling/spawn_points.txt"
            transform_info = get_spawn_points_from_file(filename)
            idx = 1157 * np.random.random_sample((12,))
            for i in range(12):
                pt = transform_info[int(idx[i])]
                # pt = transform_info[i]
                spawn_point = carla.Transform(carla.Location(x=pt[0], y=pt[1], z=pt[2]), carla.Rotation(pitch=pt[3], yaw=pt[4], roll=pt[5]))
                spawn_points.append(spawn_point)

        ## Update the cube meshes
        retcode = subprocess.call("cp random_sampling/Mesh/*_in.obj /home/siyun/CARLA/", shell=True)

        ## Spawn buildings
        batch = []
        for spawn_point in spawn_points:
            batch.append(SpawnActor(building_bp, spawn_point))

        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                building_list.append(results[i].actor_id)
        print("building list is:")
        print(building_list)

        actors = world.get_actors(building_list)
        # actors = world.get_actors().filter("building.mybuilding.cube")
        print(actors)
        for i in range (len(actors)):
            actors[i].set_building_id(i)
            actors[i].update_actor_mesh()
            print(actors[i].get_location())
        # actors[0].update_actor_mesh()

        fout = open(out_dir+"/ep_"+str(args.iter)+"/buildings.txt", 'w')
        for act in actors:
            print(act, file=fout)
        fout.close()
        fout = open(out_dir+"/ep_"+str(args.iter)+"/spawned.txt", 'w')
        for pt in spawn_points:
            print(pt, file=fout)
        fout.close()

        print('spawned %d buildings, press Ctrl+C to exit.' % (len(building_list)))

        # ## Save a copy of the cube meshes applied
        # retcode = call("cp /home/siyun/CARLA/*_in.obj data/ep_" + args.iter + "/meshes/")

        while True:
            world.wait_for_tick()
            if os.path.exists(out_dir+"/ep_"+str(args.iter)+"/with_cube/end.txt"):
                break

    finally:
        print('\ndestroying %d buildings' % len(building_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in building_list])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
