# Written by Alban Bornet, EPFL, 2021-2022
import numpy as np
import imageio
from PIL import Image, ImageDraw
from copy import deepcopy
MONITOR_REFRESH_RATE = 120
IMAGE_DIMS = (256, 256, 3)
MAX_SHAPE_SIZE = max(IMAGE_DIMS) / 2
MIN_SHAPE_SIZE = max(IMAGE_DIMS) / 20
N_FRAMES = 50


def create_random_bar_dict():
    
    h, w, n_channels = IMAGE_DIMS
    new_bar_dict = {
        'pos': np.array((h * np.random.rand(), w * np.random.rand())),  # (row, col)??
        'vpos': 5 * (np.random.rand(2) - 0.5) * 2,  # (vrow, vcol)??
        'apos': 0.5 * (np.random.rand(2) - 0.5) * 2,  # (vrow, vcol)??
        'ori': np.pi * np.random.rand(),  # sign doesn't matter
        'vori': np.pi / 10 * (np.random.rand() - 0.5) * 2,
        'dim': np.array((MAX_SHAPE_SIZE * np.random.rand(), 2 * np.random.rand())),  # (h, w)
        'col': np.random.randint(255, size=(n_channels,))}  # (r, g, b) or (grayscale,)
    
    return new_bar_dict


def create_sqm_bar_dict_list(offset_amplitude):
    
    h, w, n_channels = IMAGE_DIMS
    offset_direction = 2 * (np.random.randint(2) - 0.5)
    offset = offset_direction * offset_amplitude
    v_space = MAX_SHAPE_SIZE / 10 * np.random.rand()

    upper_bar_dict = {
        'pos': np.array((np.float(h), np.float(w))) / 2, #+ np.array((h, w)) / 1000 * np.random.rand(2,), # starting position of the sequence
        #'vpos': np.array((5 * (np.random.rand() - 0.5) * 2, 0.0)),  #  5 * (np.random.rand(2) - 0.5) * 2,
        'vpos': np.array((3.0, 0.0)), # change in position for each frame in pixels
        'apos': np.zeros((2,)), # change in diagonal position for each frame in orientation(?)
        'ori': 0.0, # orientation of line segments
        'vori': 0.0, # spin of the line segments for each frame in units of orientation
        'dim': np.array((20, 2)), # size of each of the line segments in pixels
        # 'dim': np.array((MAX_SHAPE_SIZE / 2 * np.random.rand() + MIN_SHAPE_SIZE, MAX_SHAPE_SIZE / 20 * np.random.rand())),
        'col': np.random.randint(200, 255) * np.ones((n_channels,), dtype=np.int32), # color of the line segments
        'offset': offset}

    upper_bar_dict['pos'][0] -= upper_bar_dict['dim'][1] / 2
    upper_bar_dict['pos'][1] -= upper_bar_dict['dim'][0] / 2

    lower_bar_dict = deepcopy(upper_bar_dict)
    lower_bar_dict['pos'][1] += lower_bar_dict['dim'][0] + v_space
    lower_bar_dict['offset'] *= (-1)

    upper_bar_dict_2, lower_bar_dict_2 = deepcopy(upper_bar_dict), deepcopy(lower_bar_dict)
    upper_bar_dict_2['vpos'] *= (-1)
    lower_bar_dict_2['vpos'] *= (-1)

    return [upper_bar_dict, lower_bar_dict, upper_bar_dict_2, lower_bar_dict_2]


def create_frame(bar_dict_list):

    frame = Image.fromarray(np.zeros(IMAGE_DIMS, dtype=np.uint8))
    draw = ImageDraw.Draw(frame)

    for bar_dict in bar_dict_list:
        bar_params = [bar_dict[k] for k in ['pos', 'ori', 'dim']]
        draw_coords = get_draw_coords(*bar_params)
        draw.polygon([tuple(p) for p in draw_coords], fill=tuple(bar_dict['col']))

    return np.asarray(frame)


def get_draw_coords(pos, ori, dim):

    p0 = (pos[0] + dim[0] / 2 * np.sin(ori) + dim[1] / 2 * np.cos(ori),
          pos[1] + dim[0] / 2 * np.cos(ori) + dim[1] / 2 * np.sin(ori))
    p1 = (pos[0] + dim[0] / 2 * np.sin(ori) - dim[1] / 2 * np.cos(ori),
          pos[1] + dim[0] / 2 * np.cos(ori) - dim[1] / 2 * np.sin(ori))
    p2 = (pos[0] - dim[0] / 2 * np.sin(ori) + dim[1] / 2 * np.cos(ori),
          pos[1] - dim[0] / 2 * np.cos(ori) + dim[1] / 2 * np.sin(ori))
    p3 = (pos[0] - dim[0] / 2 * np.sin(ori) - dim[1] / 2 * np.cos(ori),
          pos[1] - dim[0] / 2 * np.cos(ori) - dim[1] / 2 * np.sin(ori))

    return [p0, p2, p3, p1]


def update_bar_dict_list(bar_dict_list):

    for bar_dict in bar_dict_list:
        bar_dict['vpos'] += bar_dict['apos']
        bar_dict['pos'] += bar_dict['vpos']
        bar_dict['ori'] += bar_dict['vori']
        bar_dict['ori'] = bar_dict['ori'] % (2 * np.pi)    


def add_offset(bar_dict_list, mode='add'):
    
    for bar_dict in bar_dict_list:
        if mode == 'add':
            bar_dict['pos'][0] += bar_dict['offset']
        else:
            bar_dict['pos'][0] -= bar_dict['offset']


def create_sequence(n_frames, offset_amplitude=0.0, offset_frames=[]):
    
    # n_bars = np.random.randint(n_max_bars) + 1
    # bar_dict_list = [create_random_bar_dict() for _ in range(n_bars)]
    bar_dict_list = create_sqm_bar_dict_list(offset_amplitude)

    sequence = []
    for t in range(n_frames):
        if t - 1 in offset_frames or (t in [(-1) * f for f in offset_frames] and t > 0):
            add_offset(bar_dict_list, mode='subtract')
        if t in offset_frames or (t - 1 in [(-1) * f for f in offset_frames] and t - 1 > 0):
            add_offset(bar_dict_list)
        sequence.append(create_frame(bar_dict_list))
        update_bar_dict_list(bar_dict_list)

    return sequence


def main():

    frame_rate = 10
    frame_duration = 2 / min(frame_rate, MONITOR_REFRESH_RATE)
    # positive offset_amplitude indicates a right offset, and negative offset_amplitude indicates a left offset
    # to get the nth frame as the same offset amplitude but in the other direction, set offset_frames of the
    # desired frame to have a negative value instead.
    offset_amplitude = 2
    offset_frames = [0, 8]
    my_sequence = create_sequence(N_FRAMES, offset_amplitude=offset_amplitude, offset_frames=offset_frames)
    imageio.mimsave(f'.\hi.gif', my_sequence, duration=frame_duration)


if __name__ == '__main__':
    main()
