import os
from os.path import join

import click
import torch
import open3d as o3d
import yaml
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from tcore.datasets.fruits import IGGFruitDatasetModule
from tcore.models.model import TCoRe
import numpy as np

@click.command()
@click.option("--w", type=str, required=True)
@click.option("--bb_cr", type=float, default=None, required=False)
@click.option("--dec_cr", type=float, default=None, required=False)
@click.option("--iterative", is_flag=True)
@click.option("--model_cfg_path", type=str, default="tcore/config/model.yaml", required=False)
@click.option("--input_file_name", type=str, required=True)
@click.option("--output_file_name", type=str, default=None, required=False)
def main(w, bb_cr, dec_cr, iterative, model_cfg_path, input_file_name, output_file_name):
    model_cfg = edict(yaml.safe_load(open(model_cfg_path)))
    backbone_cfg = edict(
        yaml.safe_load(open("tcore/config/backbone.yaml"))
    )
    decoder_cfg = edict(
        yaml.safe_load(open("tcore/config/decoder.yaml"))
    )
    cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})

    cfg.EVALUATE = True
    cfg.MODEL.OVERFIT = True

    if bb_cr:
        cfg.BACKBONE.CR = bb_cr
    if dec_cr:
        cfg.DECODER.CR = dec_cr

    if iterative:
        cfg.DECODER.ITERATIVE_TEMPLATE = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = TCoRe(cfg).to(device)
    w = torch.load(w, map_location = device)
    model.load_state_dict(w["state_dict"], strict=False)

    model.eval()

    input_pcd = o3d.io.read_point_cloud(input_file_name)

    # # downsample the point cloud
    # input_pcd = input_pcd.voxel_down_sample(voxel_size=0.001)
    # # move the points to the center
    # input_pcd.points = o3d.utility.Vector3dVector(np.array(input_pcd.points) - np.mean(np.array(input_pcd.points), axis=0))
    # # change scale to 2.0
    # input_pcd.scale(1/np.max(np.linalg.norm(np.array(input_pcd.points), axis=1), axis=0) * 0.04, center=np.array([0, 0, 0]))
    # # filter out the points x < 0
    # input_pcd = input_pcd.select_by_index(np.where(np.array(input_pcd.points)[:, 0] > 0)[0])
   

    input_points = torch.tensor(input_pcd.points, dtype=torch.float32, device='cpu')
    input_colors = torch.tensor(input_pcd.colors, dtype=torch.float32, device='cpu')
    x = {'points': input_points[None, ...], 'colors': input_colors[None, ...]}

    outputs = model.forward(x)
    meshes = model.get_meshes(outputs)
    output_mesh = meshes[0]
    output_mesh = output_mesh.filter_smooth_taubin(10)
    output_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([input_pcd, output_mesh])

    if output_file_name is not None:
        o3d.io.write_triangle_mesh(output_file_name, output_mesh)


if __name__ == "__main__":
    main()
