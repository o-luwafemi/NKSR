# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import nksr
import torch

from pycg import vis, exp
from pathlib import Path
import numpy as np
from common import load_scannet_example, load_eagle, load_olympic_flame


if __name__ == '__main__':
    device = torch.device("cuda:0")

    scannet_geom = load_scannet_example()
    
    eagle_geom = load_eagle()
    olympic_geom = load_olympic_flame()

    input_xyz = torch.from_numpy(np.asarray(olympic_geom.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(olympic_geom.normals)).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")
    
    field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=0.02, chunk_size=20.0)
    mesh = field.extract_dual_mesh(mise_iter=2)

    vis.show_3d([vis.mesh(mesh.v, mesh.f)], [olympic_geom])
