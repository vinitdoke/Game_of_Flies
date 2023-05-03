import os
import time
from argparse import ArgumentParser

import numpy as np
from vizman import Visualiser
import imageio
from tqdm import tqdm
from vispy import app


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="path to directory of simulation"
    )
    parser.add_argument(
        "-v",
        "--visual",
        action="store_true",
        default=False,
        help="Visualise in UI (post sim)",
    )

    return parser.parse_args()


def load_data(path):
    data = np.load(path)
    return data


def vidwriter(dirpath, limits, type_array):
    resolution = (4 * 800, 4 * 600)
    # resolution = (1980, 1080)
    visual = Visualiser(size=resolution)
    # visual.canvas.size = (1920, 1080)
    # visual.canvas.dpi = 500
    visual.init_plotting(limits, type_array)
    visual.draw_boundary(limits)

    output_filename = os.path.join(dirpath, "output.mp4")
    writer = imageio.get_writer(output_filename, fps=30)

    # visual.view.camera.set_range(x=[0, 100])
    # total files in frames directory
    list_of_frames = os.listdir(os.path.join(dirpath, "frames"))
    total_files = len(list_of_frames)

    for frame in tqdm(list_of_frames):
        data = load_data(os.path.join(dirpath, "frames", frame))
        visual.blind_update(data)
        writer.append_data(visual.get_render(alpha=False))

    writer.close()


def liveviz(dirpath, limits, type_array):
    resolution = (2 * 800, 2 * 600)

    visual = Visualiser(size=resolution, filepath=dirpath)
    visual.init_plotting(limits, type_array)
    visual.draw_boundary(limits)
    visual.start()
    app.run()


if __name__ == "__main__":

    args = parse()
    if args.input is None:
        raise ValueError("No input path specified")

    state_data = np.load(os.path.join(args.input, "state.npz"))
    type_array = state_data["type_array"]
    num_particles = state_data["num_particles"]
    type_array = type_array[:num_particles]
    limits = state_data["limits"]

    if args.visual:
        liveviz(args.input, limits, type_array)
    else:
        vidwriter(args.input, limits, type_array)

    # data = load_data(args.input)
