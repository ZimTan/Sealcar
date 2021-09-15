import argparse

def argparser():
    # We create the argument parser
    parser = argparse.ArgumentParser()

    # We create a choice argument (because we want either xbox, keras or 
    # keyboard but not at the same time)
    parser.add_argument('-m', '--mode', required=True,
            choices=['xbox', 'keras', 'keyboard', 'direction'],
            help='Choose the controller (xbox, keras, keyboard or direction)')

    args = parser.parse_args()

    return args
