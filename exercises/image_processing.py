from PIL import Image
from itertools import product
import multiprocessing as mp
import numpy as np
import sys

if __name__ == "__main__":
    resize_factor = 9.876

    number_of_cores = int(sys.argv[1])

    for i in range(1, int(np.sqrt(number_of_cores)) + 1):
        if number_of_cores % i == 0:
            number_of_columns = i
            number_of_rows = number_of_cores // i

    with Image.open('globe_east_lrg.jpg', 'r') as image:
        img = np.array(image).astype(float)
    original_image_shape = img.shape
    resized_image_shape = (int(img.shape[0] // resize_factor),
                           int(img.shape[1] // resize_factor),
                           img.shape[2])
    image_canvas = np.zeros(resized_image_shape, dtype=np.uint8)

    worker_rows = np.array_split(np.arange(resized_image_shape[0]), number_of_rows)
    worker_columns = np.array_split(np.arange(resized_image_shape[1]), number_of_columns)
    np.save('output_file.npy', image_canvas)

    def get_original_index(i, j):
        return (i * resize_factor, j * resize_factor)

    def get_bilinear_arguments(i, j):
        i_orig, j_orig = get_original_index(i, j)
        i_orig_floor = int(i_orig)
        j_orig_floor = int(j_orig)
        i_distance = i_orig - i_orig_floor
        j_distance = j_orig - j_orig_floor
        factors_i = (1 - i_distance, i_distance)
        factors_j = (1 - j_distance, j_distance)
        composite_factors = (factors_i[0] * factors_j[0],
                             factors_i[0] * factors_j[1],
                             factors_i[1] * factors_j[0],
                             factors_i[1] * factors_j[1])
        discrete_indices = ((i_orig_floor, j_orig_floor),
                            (i_orig_floor, j_orig_floor + 1),
                            (i_orig_floor + 1, j_orig_floor),
                            (i_orig_floor + 1, j_orig_floor + 1))
        for num in range(4):
            for dim in range(2):
                if discrete_indices[num][dim] >= original_image_shape[dim]:
                    discrete_indices[num][dim] = original_image_shape[dim] - 1
            yield discrete_indices[num], composite_factors[num]

    def get_interpolated_value(i, j):
        value = np.zeros((3,))
        for index, weight in get_bilinear_arguments(i, j):
            value += img[*index] * weight
        return value

    def interpolate_and_write(i, j, queue):
        value = get_interpolated_value(i, j)
        # queue.put((i, j, value))
        return value

    def interpolate_over_indices(irange, jrange, queue):
        out_map = np.zeros((len(irange), len(jrange), 3), dtype=np.uint8)
        for i, j in product(irange, jrange):
            out_map[i - irange[0], j - jrange[0]] = interpolate_and_write(i, j, queue)
        queue.put((slice(irange[0], irange[-1] + 1),
                  slice(jrange[0], jrange[-1] + 1),
                  out_map))

    def queue_writer(queue):
        while True:
            next = queue.get()
            if next == 'END':
                break
            output = np.load('output_file.npy')
            output[next[0], next[1]] = next[2]
            np.save('output_file.npy', output)

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(processes=(number_of_cores + 1))
    qw = pool.apply_async(queue_writer, (queue,))
    jobs = []
    for (r, c) in product(worker_rows, worker_columns):
        job = pool.apply_async(interpolate_over_indices, (r, c, queue))
        jobs.append(job)
    for job in jobs:
        job.get()
    queue.put('END')
    qw.get()
    out_img = Image.fromarray(np.load('output_file.npy'))
    out_img.show()
    out_img.save('output_file.png')
