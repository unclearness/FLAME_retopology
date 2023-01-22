import obj_io
import FLAME_util
import torch

if __name__ == '__main__':
    flame, flame_config = FLAME_util.getFlame()
    neutral_shape_params = torch.zeros(
        (flame.batch_size, flame_config.shape_params))
    neutral_expression_params = torch.zeros(
        (flame.batch_size, flame_config.expression_params))
    neutral_pose_params = torch.zeros(
        (flame.batch_size, 6))
    vertices, landmarks = flame(
        neutral_shape_params, neutral_expression_params, neutral_pose_params)
    obj_io.saveObjSimple("./data/neutral_flame.obj", vertices.to('cpu').detach(
    ).numpy().copy()[0], flame.faces)
