import time

import cv2
import numpy as np

STEP = 0.01
NSTEPS = 20

ITERATIONS = 30
EPSILON = 1e-6

WIDTH = 640
HEIGHT = 480

XMIN = -2.5
XMAX = 1
XDOMSIZE = XMAX - XMIN
YMIN = -1
YMAX = 1
YDOMSIZE = YMAX - YMIN

ROOTS = np.asarray(
    [1, -0.5 + (np.sqrt(3) / 2) * 1j, -0.5 + -(np.sqrt(3) / 2) * 1j],
    dtype=np.complex64
)

BLACK = np.zeros(3, dtype=np.uint8)
WHITE = np.ones(3, dtype=np.uint8) * 255
COLOURS = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

def fractal_image(k):
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            z = XMIN + XDOMSIZE * x / WIDTH + 1j * (YMIN + YDOMSIZE * y / HEIGHT)
            for _ in range(ITERATIONS):
                z -= (z ** k - 1) / (k * z ** (k - 1)) # f(z) / f'(z)

            for i, root in enumerate(ROOTS):
                delta = z - root
                if abs(delta) < EPSILON:
                    image[y][x] = COLOURS[i]
                    break
            else:
                image[y][x] = BLACK

    return image

def boundaries(image):
    binary = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    for y in range(1, HEIGHT - 1):
        for x in range(1, WIDTH - 1):
            colour = image[y][x]
            surrounding = np.reshape(image[y - 1:y + 2, x - 1:x + 2], (9, 3))

            if not all(np.array_equal(pixel, colour) for pixel in surrounding):
                binary[y][x] = WHITE
    
    return binary

THRESHOLD = 1
THRESHOLD_MULTIPLIER = 3
KERNEL_SIZE = 3

def canny_boundaries(image):
    return cv2.Canny(image, THRESHOLD, THRESHOLD * THRESHOLD_MULTIPLIER, KERNEL_SIZE)

KERNEL = np.asarray([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

def rgb_to_binary(image):
    return cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY
    )[1] # (threshold, image)

def filter_boundaries(image):
    return rgb_to_binary(cv2.filter2D(image, -1, KERNEL))

for i in range(NSTEPS):
    start = time.time()
    image = fractal_image(3 + STEP * i)
    print(f"image {i} creation: {round(time.time() - start, 2)}s")

    start = time.time()
    binary = filter_boundaries(image)
    print(f"filter {i} boundaries: {round(time.time() - start, 2)}s")

    cv2.imwrite(f"out{i}.png", binary)
