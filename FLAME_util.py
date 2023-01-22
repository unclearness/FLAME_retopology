from FLAME_PyTorch import FLAME, config


def getFlame():
    my_conifg = config.get_config()
    my_conifg.batch_size = 1
    my_conifg.shape_params = 300
    my_conifg.expression_params = 100
    my_conifg.flame_model_path = './model/FLAME2019/generic_model.pkl'
    my_conifg.static_landmark_embedding_path = './model/FLAME2020/flame_static_embedding.pkl'
    my_conifg.dynamic_landmark_embedding_path = './model/FLAME2020/flame_dynamic_embedding.npy'
    flame = FLAME.FLAME(my_conifg)
    return flame, my_conifg


if __name__ == '__main__':
    getFlame()
