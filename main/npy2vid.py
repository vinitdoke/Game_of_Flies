import os
from argparse import ArgumentParser

import numpy as np
from vizman import Visualiser
import imageio
from tqdm import tqdm


def parse():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='path to directory of simulation')

    return parser.parse_args()

def load_data(path):
    data = np.load(path)
    return data

def main(dirpath, limits, type_array):
    resolution = (2*800, 2*600)
    # resolution = (1980, 1080)
    visual = Visualiser(size=resolution)
    # visual.canvas.size = (1920, 1080)
    # visual.canvas.dpi = 500
    visual.init_plotting(limits, type_array)
    visual.draw_boundary(limits)

    output_filename = os.path.join(dirpath, "output.mp4")
    writer = imageio.get_writer(output_filename, fps=15)

    # visual.view.camera.set_range(x=[0, 100])
    # total files in frames directory
    list_of_frames = os.listdir(os.path.join(dirpath, "frames"))
    total_files = len(list_of_frames)
    
    for frame in tqdm(list_of_frames):
        data = load_data(os.path.join(dirpath, "frames",frame))
        visual.blind_update(data)
        writer.append_data(visual.get_render(alpha=False))

    writer.close()


if __name__ == "__main__":

    args = parse()
    if args.input is None:
        raise ValueError("No input path specified")

    state_data = np.load(os.path.join(args.input, "state.npz"))
    type_array = state_data["type_array"]
    num_particles = state_data["num_particles"]
    type_array = type_array[:num_particles]
    limits = state_data["limits"]

    main(args.input, limits, type_array)

 
    # data = load_data(args.input)
