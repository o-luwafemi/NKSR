# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
import nksr
import open3d as o3d
from pycg import vis, exp
from common import load_bunny_example, load_eagle, load_olympic_flame, load_pix4dmatic, warning_on_low_memory

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    warning_on_low_memory(1024.0)
    device = torch.device("cuda:0")

    bunny_geom = load_bunny_example()
    eagle_geom = load_eagle()
    olympic_geom = load_olympic_flame()
    pix4dmatic_geom = load_pix4dmatic().voxel_down_sample(voxel_size=0.1)
    # downpcd = pix4dmatic_geom.voxel_down_sample(voxel_size=0.05)

    input_xyz = torch.from_numpy(np.asarray(eagle_geom.points)).float().to(device)
    input_normal = torch.from_numpy(np.asarray(eagle_geom.normals)).float().to(device)

    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")
    
    field = reconstructor.reconstruct(input_xyz, input_normal,
                                      detail_level = 0.74,
                                      # chunk_size = 50.0,
                                      # voxel_size=0.01,
                                      )
    
    mesh = field.extract_dual_mesh(mise_iter=1)
    # o3d.io.write_triangle_mesh("eagle_3d.ply", mesh)

    vis.show_3d([vis.mesh(mesh.v, mesh.f)], [eagle_geom])
    
    # exp.save_3d([vis.mesh(mesh.v, mesh.f)], [eagle_geom])



