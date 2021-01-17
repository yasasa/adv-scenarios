import numpy as np
import logging
from .optimizer import CubeOptimizer
from scripts.buffered_saver import TrajectorySaver
import os
from shapely import geometry
import math

# for debug
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt


class CEM(CubeOptimizer):

    CUBE_PARAM_LEN = 7
    NUM_CUBE_LOC_DIM = 3
    NUM_CUBE_ROT_DIM = 1
    NUM_CUBE_SCALE_DIM = 3

    def __init__(self,
                 dynamics,
                 sensor,
                 policy,
                 step_cost,
                 save_path,
                 elite_frac=0.2,
                 state_dim=3,
                 action_dim=1,
                 cube_dim=2,
                 integration_timesteps=200,
                 log_file_name='CEM_output.log',
                 config=None):

        super(CEM, self).__init__(dynamics, sensor, policy, step_cost, save_path, state_dim, action_dim, cube_dim)

        self._elite_frac = config['elite_frac']
        self._T = integration_timesteps
        # self._trajectory_saver = TrajectorySaver(os.path.join(save_path, 'Trajectory'), integration_timesteps, 3)
        log_file_handler = logging.FileHandler(os.path.join(save_path, log_file_name))
        log_file_handler.setLevel(logging.INFO)
        self.log().addHandler(log_file_handler)
        self._initial_std = np.array(config['initial_cube_std'], dtype=np.float32)
        self._epsilon = np.array(config['epsilon'], dtype=np.float32)
        self._iterations = config['cem_iter']
        self._population = config['cem_population']
        self._fixed_loc = config['fixed_loc']
        self._fixed_rot = config['fixed_rot']
        self._fixed_sca = config['fixed_sca']
        self._feasible_region = geometry.Polygon([[94., 100.], [94., 123.], [94.5, 124.4], [95.3, 125.6], 
                                                    [96.8, 126.5], [98.1, 126.7], [99.6, 127.2], [100.5, 127.8],
                                                    [120., 127.8], [120., 100.]])

    def cem_evaluate(self, c_xy, c_rot, c_scale, x0, num_new_cubes):
        c_xy = c_xy.reshape(num_new_cubes, -1)
        c_rot = c_rot.reshape(num_new_cubes, -1)
        c_scale = c_scale.reshape(num_new_cubes, -1)
        x = x0[:]
        c = np.array([], dtype=np.float32)
        for num_cube in range(num_new_cubes):
            cube_param = np.append(np.append(c_xy[num_cube], c_rot[num_cube]), c_scale[num_cube])
            c = np.concatenate((c, cube_param))
            self.log("cem_evaluate").info("Cube {} Parameters {}".format(num_cube, cube_param))

        total_cost = 0
        for i in range(self._T):
            u = self._policy(self._sensor(x, c))
            x = self._dynamics(x, u)
            self.save('Trajectory.npy', np.array(x, dtype=np.float32))
            total_cost += self._step_cost.cost(x, u)

        self.log("cem_evaluate").info("Total Cost: %.3f" % (total_cost))
        self.save('Trajectory_cost.npy', np.array([total_cost], dtype=np.float32))
        return total_cost

    def _get_offsets(self, sc_x, sc_y, yaw):
        original_corners = [(-sc_x/2, -sc_y/2), (sc_x/2, -sc_y/2), (sc_x/2, sc_y/2), (-sc_x/2, sc_y/2)]
        ret = []
        for x, y in original_corners:
            x_new = math.cos(yaw) * x - math.sin(yaw) * y
            y_new = math.sin(yaw) * x + math.cos(yaw) * y
            ret += [[x_new, y_new]]
        return ret


    def run(self, c0_np, x0):
        begin_eps, end_eps = self._epsilon
        if self._iterations <= 1:
            delta_eps = 0
        else:
            delta_eps = (begin_eps - end_eps)/(self._iterations-1)

        num_elite = max(int(self._elite_frac * self._population), 1)

        num_new_cubes = c0_np.shape[0]

        # initialize mean and standard deviation
        cubes_loc_mean = np.zeros((num_new_cubes, self.NUM_CUBE_LOC_DIM))
        cubes_rot_mean = np.zeros((num_new_cubes, self.NUM_CUBE_ROT_DIM))
        cubes_scale_mean = np.ones((num_new_cubes, self.NUM_CUBE_SCALE_DIM))

        cubes_loc_std = self._initial_std[0]*np.ones((num_new_cubes, self.NUM_CUBE_LOC_DIM))
        cubes_rot_std = self._initial_std[1]*np.ones((num_new_cubes, self.NUM_CUBE_ROT_DIM))
        cubes_scale_std = self._initial_std[2]*np.ones((num_new_cubes, self.NUM_CUBE_SCALE_DIM))

        for num_cube in range(num_new_cubes):
            cubes_loc_mean[num_cube] = c0_np[num_cube][0:self.NUM_CUBE_LOC_DIM]
            cubes_rot_mean[num_cube] = c0_np[num_cube][self.NUM_CUBE_LOC_DIM:self.NUM_CUBE_LOC_DIM+self.NUM_CUBE_ROT_DIM]
            cubes_scale_mean[num_cube] = c0_np[num_cube][self.NUM_CUBE_LOC_DIM+self.NUM_CUBE_ROT_DIM:]

        for itr in range(self._iterations):
            eps = begin_eps - delta_eps*itr
            self.log().info("Iteration %i" % itr)
            self.log().info("Epsilon: %.3f" % eps)
            locs_current_iter = np.empty((self._population, 0))
            rots_current_iter = np.empty((self._population, 0))
            scales_current_iter = np.empty((self._population, 0))

            for num_cube in range(num_new_cubes):
                self.log().info("Cube %d xyz mean: (%.3f, %.3f, %.3f) std: (%.3f, %.3f, %.3f)" % (num_cube, cubes_loc_mean[num_cube][0], cubes_loc_mean[num_cube][1], cubes_loc_mean[num_cube][2], cubes_loc_std[num_cube][0], cubes_loc_std[num_cube][1], cubes_loc_std[num_cube][2]))
                self.log().info("Cube %d rot mean: (%.3f) std: (%.3f)" % (num_cube, cubes_rot_mean[num_cube,0], cubes_rot_std[num_cube,0]))
                self.log().info("Cube %d scale mean: (%.3f, %.3f, %.3f) std: (%.3f, %.3f, %.3f)" % (num_cube, cubes_scale_mean[num_cube,0], cubes_scale_mean[num_cube,1], cubes_scale_mean[num_cube,2], cubes_scale_std[num_cube,0], cubes_scale_std[num_cube, 1], cubes_scale_std[num_cube,2]))

                if not self._fixed_loc:
                    locs = np.random.multivariate_normal(mean=cubes_loc_mean[num_cube], cov=np.diag(cubes_loc_std[num_cube]**2+eps), size=self._population)         # (50, 2)  --> ignore z
                else:
                    locs = np.random.multivariate_normal(mean=cubes_loc_mean[num_cube], cov=np.diag(cubes_loc_std[num_cube]*0), size=self._population)

                if not self._fixed_rot:
                    rots = np.random.multivariate_normal(mean=cubes_rot_mean[num_cube], cov=np.diag(cubes_rot_std[num_cube]**2+eps), size=self._population)                 # (50, 1)
                else:
                    rots = np.random.multivariate_normal(mean=cubes_rot_mean[num_cube], cov=np.diag(cubes_rot_std[num_cube]*0), size=self._population)

                if not self._fixed_sca:
                    scales = np.random.multivariate_normal(mean=cubes_scale_mean[num_cube], cov=np.diag(cubes_scale_std[num_cube]**2+eps), size=self._population)           # (50, 3)
                else:
                    scales = np.random.multivariate_normal(mean=cubes_scale_mean[num_cube], cov=np.diag(cubes_scale_std[num_cube]*0), size=self._population)

                for i in range(self._population):
                    cx, cy, cz = locs[i]
                    yaw = rots[i][0]
                    sc_x, sc_y, sc_z = scales[i]
    
                    while sc_x < 0. or sc_y < 0. or sc_z < 0. or sc_x > 3. or sc_y > 3. or sc_z > 3.:
                        sc_x, sc_y, sc_z = np.random.multivariate_normal(mean=cubes_scale_mean[num_cube], cov=np.diag(cubes_scale_std[num_cube]**2), size=1)[0]
                    scales[i] = np.array([sc_x, sc_y, sc_z], dtype=np.float32)

                    # check for infeasible cube parameters
                    # new implementation
                    cube_bound_offset = self._get_offsets(sc_x, sc_y, yaw)
                    cube_bound = geometry.Polygon([ [x+cx, y+cy] for x, y in cube_bound_offset ])

                    while cz < 0.1 or cz > 0.3:
                        cz = np.random.normal(loc=cubes_loc_mean[num_cube][-1], scale=cubes_loc_std[num_cube][-1]**2, size=1)

                    while not self._feasible_region.contains(cube_bound):
                        cx, cy = np.random.multivariate_normal(mean=cubes_loc_mean[num_cube][:-1], cov=np.diag(cubes_loc_std[num_cube][:-1]**2), size=1)[0]
                        cube_bound = geometry.Polygon([ [x+cx, y+cy] for x, y in cube_bound_offset ])

                    locs[i] = np.array([cx, cy, cz], dtype=np.float32)


                locs_current_iter = np.concatenate((locs_current_iter, locs), axis=1)
                rots_current_iter = np.concatenate((rots_current_iter, rots), axis=1)
                scales_current_iter = np.concatenate((scales_current_iter, scales), axis=1)

            locs_current_iter = np.concatenate((cubes_loc_mean.reshape(1,-1), locs_current_iter), axis=0)
            rots_current_iter = np.concatenate((cubes_rot_mean.reshape(1,-1), rots_current_iter), axis=0)
            scales_current_iter = np.concatenate((cubes_scale_mean.reshape(1,-1), scales_current_iter), axis=0)

            f = lambda cube_xy, cube_rot, cube_scale: self.cem_evaluate(cube_xy, cube_rot, cube_scale, x0, num_new_cubes)
            rewards = np.fromiter(map(f, locs_current_iter, rots_current_iter, scales_current_iter), dtype='float64')

            # Get elite parameters
            elite_inds = rewards.argsort()[::-1][:num_elite]
            elite_locs = locs_current_iter[elite_inds]
            elite_rots = rots_current_iter[elite_inds]
            elite_scales = scales_current_iter[elite_inds]

            # Update theta_mean and theta_std
            cubes_loc_mean = np.mean(elite_locs, axis=0).reshape(num_new_cubes, -1)
            cubes_loc_std = np.std(elite_locs, axis=0).reshape(num_new_cubes, -1)
            cubes_rot_mean = np.mean(elite_rots, axis=0).reshape(num_new_cubes, -1)
            cubes_rot_std = np.std(elite_rots, axis=0).reshape(num_new_cubes, -1)
            cubes_scale_mean = np.mean(elite_scales, axis=0).reshape(num_new_cubes, -1)
            cubes_scale_std = np.std(elite_scales, axis=0).reshape(num_new_cubes, -1)

            self.log().info("Iteration %i. Min f: %8.3g. Mean f: %8.3g. Max f: %8.3g" % (itr, np.min(rewards), np.mean(rewards), np.max(rewards)))

        return cubes_loc_mean
