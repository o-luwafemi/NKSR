import torch
import nksr
import numpy as np
from pycg import vis
from common import load_bunny_example, warning_on_low_memory

bunny_geom = load_bunny_example()

input_xyz = torch.from_numpy(np.asarray(bunny_geom.points)).float().to(device)
input_normal = torch.from_numpy(np.asarray(bunny_geom.normals)).float().to(device)

reconstructor = nksr.Reconstructor(device)
field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=1.0)
mesh = field.extract_dual_mesh(mise_iter=1)