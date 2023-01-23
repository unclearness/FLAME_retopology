from FLAME_PyTorch import FLAME, config


def getFlame():
    my_config = config.get_config()
    my_config.batch_size = 1
    my_config.shape_params = 300
    my_config.expression_params = 100
    # 2020 version has slightly better expressions
    my_config.flame_model_path = './model/FLAME2020/generic_model.pkl'
    # TODO: 2020 and 2019 version mismatch. But it works since landmarks are not used in this project
    my_config.static_landmark_embedding_path = './model/FLAME2019/flame_static_embedding.pkl'
    my_config.dynamic_landmark_embedding_path = None
    my_config.use_face_contour = False
    flame = FLAME.FLAME(my_config)
    return flame, my_config


if __name__ == '__main__':
    getFlame()
