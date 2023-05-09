# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import torch
import math

from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *
from math import sqrt

# terrain generator
class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.horizontal_scale = 0.3 #  0.1
        self.vertical_scale = 0.1  # 0.005
        self.border_size = 3.0  # 20
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        # self.proportions = [np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg["terrainProportions"]))]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        # self.num_maps = self.env_rows * self.env_cols
        # self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)

        
        self.curriculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale)

    def curriculum(self, num_robots, num_terrains, num_levels):

        for j in range(num_terrains):
            for i in range(num_levels):
    
                terrain = SubTerrain("terrain",
                                     width=self.width_per_env_pixels,
                                     length=self.length_per_env_pixels,
                                     vertical_scale=self.vertical_scale,
                                     horizontal_scale=self.horizontal_scale)

                if j == 0:
                    self.box_obstacle(i, terrain)
                
                if j == 1:
                    self.box_obstacle(i, terrain)
                    # self.circle_box_obstacle(i, terrain)
                    
                # Heightfield coordinate system
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw                

                env_origin_x = (i + 0.5) * self.env_length  # meters
                env_origin_y = (j + 0.5) * self.env_width  # meters
                x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale  # meters
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]  # meters




    def box_obstacle(self, diff_level, terrain):
        cube_width = 1.5  # meters
        cube_length = 1.5  # meters
        cube_height = 0.5 + diff_level * 0.3  # meters
        c_width, c_length = self.width_per_env_pixels // 2,  self.length_per_env_pixels // 2
        cube_width_pixels = int(cube_width / self.horizontal_scale)
        cube_length_pixels = int(cube_length / self.horizontal_scale)
        terrain.height_field_raw[c_length - cube_length_pixels // 2: c_length + cube_length_pixels // 2 + 1,
                                 c_width - cube_width_pixels // 2: c_width + cube_width_pixels // 2 + 1, ] = int(cube_height / self.vertical_scale)


    def circle_box_obstacle(self, diff_level, terrain):
        circle_cube_width = 2
        circle_cube_length = 2
        c_width, c_length = self.width_per_env_pixels // 2,  self.length_per_env_pixels // 2
     
        circle_cube_width_pixels = int(circle_cube_width / self.horizontal_scale)
        circle_cube_length_pixels = int(circle_cube_length / self.horizontal_scale)

        
        # create a 2D grid of the same size as the height field
        height_field_shape = terrain.height_field_raw[c_length - circle_cube_length_pixels // 2: c_length + circle_cube_length_pixels // 2 + 1,
                     c_width - circle_cube_width_pixels // 2: c_width + circle_cube_width_pixels // 2 + 1, ].shape

        x, y = np.meshgrid(np.arange(height_field_shape[1]), np.arange(height_field_shape[0]))
        x -= circle_cube_width_pixels // 2
        y -= circle_cube_length_pixels // 2
        
        # compute the distance of each point to the center of the circle
        dist = np.sqrt(x**2 + y**2)
        
        # use a radial function to map the distance to a height value
        circle_center_height = 0.5 + diff_level * 0.3
        circle_edge_height = 0 + diff_level * 0.3
        circle_radius_pixels = int(circle_cube_width_pixels / 2)

        
        height = np.where(dist > circle_radius_pixels, 0, 
                np.add(circle_edge_height, np.multiply(np.subtract(circle_center_height, circle_edge_height), np.sqrt(np.max(np.subtract(1, np.square(dist / circle_radius_pixels)), 0))))) 

 
        # set the height of the pixels within the circle
        height /= self.vertical_scale
        terrain.height_field_raw[c_length - circle_cube_length_pixels // 2: c_length + circle_cube_length_pixels // 2 + 1,
                     c_width - circle_cube_width_pixels // 2: c_width + circle_cube_width_pixels // 2 + 1, ] = height



