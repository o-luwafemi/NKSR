import nksr
import torch
import numpy as np
from pycg import vis
from common import load_bunny_example, warning_on_low_memory


warning_on_low_memory(1024.0)
bunny_geom = load_bunny_example()
device = torch.device("cuda:0")
reconstructor = nksr.Reconstructor(device)

input_xyz = torch.from_numpy(np.asarray(bunny_geom.points)).float().to(device)
input_normal = torch.from_numpy(np.asarray(bunny_geom.normals)).float().to(device)
input_color = torch.from_numpy(np.asarray(bunny_geom.colors)).float().to(device)

# Note that input_xyz and input_normal are torch tensors of shape [N, 3] and [N, 3] respectively.
field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=1.0)
# input_color is also a tensor of shape [N, 3]
# field.set_texture_field(nksr.fields.PCNNField(input_xyz, input_color))
# Increase the dual mesh's resolution.
mesh = field.extract_dual_mesh(mise_iter=1)

# Visualizing
vis.show_3d([vis.mesh(mesh.v, mesh.f, color=mesh.c)])