import numpy as np
import obj_io
import FLAME_util
import json
import torch
import torch.nn as nn

my_device = 'cuda'


def computeInterpolated(vertices,  # (B, V, 3)
                        faces,     # (F, 3)
                        fids,      # (N)
                        barys      # (N, 2)
                        ):
    fetched = vertices[:, faces[fids]]
    v0 = fetched[:, :, 0]
    v1 = fetched[:, :, 1]
    v2 = fetched[:, :, 2]
    # Add batch
    barys = barys[None, ]
    return barys[..., 0:1] * v0 + barys[..., 1:2] * v1 + barys[..., 2:3] * v2


if __name__ == '__main__':
    corresp_path = './data/corresp_all_info.json'
    corresp_fids = []
    corresp_barys = []
    with open(corresp_path, 'r') as fp:
        corresp_raw = json.load(fp)
    for c in corresp_raw:
        if c is None:
            fid, u, v = -1, 999.9, 999.9
        else:
            fid = int(c[0])
            u = float(c[1])
            v = float(c[2])
            w = 1 - u - v
        corresp_fids.append(fid)
        corresp_barys.append([u, v, w])

    corresp_fids = torch.tensor(corresp_fids, dtype=int, device=my_device)
    corresp_barys = torch.tensor(corresp_barys, device=my_device)

    flame, flame_config = FLAME_util.getFlame()
    flame.to(my_device)
    shape_params = torch.zeros(
        (flame.batch_size, flame_config.shape_params), device=my_device)
    expression_params = torch.zeros(
        (flame.batch_size, flame_config.expression_params), device=my_device)
    pose_params = torch.zeros((flame.batch_size, 6), device=my_device)

    shape_params = nn.Parameter(shape_params)
    expression_params = nn.Parameter(expression_params)
    pose_params = nn.Parameter(pose_params)

    dst_obj_path = './data/morphed_iphone2.obj'
    dst = obj_io.loadObj(dst_obj_path)

    # Add batch
    dst_verts = torch.tensor(dst.verts[None, ], device=my_device)

    flame_faces = torch.tensor(flame.faces.astype(np.int64), device=my_device)

    optimizer = torch.optim.Adam(
        [shape_params], lr=0.01)

    max_iter = 200
    shape_reg_w = 1e-3
    exp_reg_w = 1e-5

    def exec_optim(optimizer, max_iter):
        for i in range(max_iter):
            optimizer.zero_grad()
            losses = 0
            vertices, landmarks = flame(
                shape_params, expression_params, pose_params)
            interpolated = computeInterpolated(
                vertices, flame_faces, corresp_fids, corresp_barys)

            vert_diffs = interpolated - dst_verts
            loss = torch.sum(vert_diffs * vert_diffs)
            shape_reg = torch.sum(shape_params * shape_params) * shape_reg_w
            exp_reg = torch.sum(expression_params *
                                expression_params) * exp_reg_w
            losses = loss + shape_reg + exp_reg
            losses.backward()
            optimizer.step()

            print(i, loss, shape_reg, exp_reg, losses)

    exec_optim(optimizer, 300)
    optimizer = torch.optim.Adam(
        [expression_params], lr=0.01)
    exec_optim(optimizer, 300)
    optimizer = torch.optim.Adam(
        [shape_params, expression_params], lr=0.01)
    exec_optim(optimizer, 600)

    vertices, landmarks = flame(
        shape_params, expression_params, pose_params)
    interpolated = computeInterpolated(
        vertices, flame_faces, corresp_fids, corresp_barys)
    print('Optimized')
    print('shape_params', shape_params)
    print('expression_params', expression_params)
    print('pose_params', pose_params)
    obj_io.saveObjSimple("./data/morphed_flame.obj", vertices.to('cpu').detach(
    ).numpy().copy()[0], flame.faces)

    obj_io.saveObjSimple("./data/morphed_correspondences.obj", interpolated.to('cpu').detach(
    ).numpy().copy()[0], [])
