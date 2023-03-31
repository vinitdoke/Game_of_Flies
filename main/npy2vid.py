import numpy as np
from argparse import ArgumentParser
import os


def parse():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=DATA_PATH)

    return parser.parse_args()

def load_data(path):
    data = np.load(path)
    return data


    # data = load_data(args.input)
