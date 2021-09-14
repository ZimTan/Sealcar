import argparse

def argparser():
    # We create the argument parser
    parser = argparse.ArgumentParser()

    # Adds the mandatory '-p' argument, to register the model path
    parser.add_argument('-p', '--path', required=True,
            help='the path to the model folder to load')

    # Adds the '-c' argument, to register if a new model is to be created
    parser.add_argument('-c', '--create', action='store_true',
            help='defines if a new model is to be created')

    # Adds the '-t' argument to activate multithreading
    parser.add_argument('-t', '--thread', help='activate multi-threading',
            action='store_true')

    # We parse the arguments
    args = parser.parse_args()

    return args
