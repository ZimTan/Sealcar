# Configuration file for the autocar controller:

# controller config:
controller_config = {
    'dead_zone': 0.15,
    'fix_steer': 0.0,
    'path': 'src/pytorch/models'
}

# viewer config:
viewer_config = {
    'screen_size': (640, 480)
}

# recorder config:
recorder_config = {
    'path': 'dataset/rrl1/'
}

# car config:
car_config = {
    # 'host': 'sim.diyrobocars.fr',
    'host': 'localhost',
    #'host': '192.168.103.69',
    'port': 9091,

    'car_name': None,
    'font_size': None,
    'body_style': None,
    'body_rgb': None
}
