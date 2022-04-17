import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    '''transforms a RGB image into a grayscale image'''
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def hist_eq(img):
    '''equalizes the histogram of an image'''
    img_hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = img_hist.cumsum()
    cdf_normalized = cdf * img.max() / cdf.max()
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    img_equalized = img_equalized.reshape(img.shape)
    return img_equalized


def erosion(src: np.array, kernel_size: int):
    """ erosion operation.
    .. math::
        dst(x,y) = \min_{(x', y') : kernel(x', y') \\neq 0} src(x + x', y + y')
    """
    # Credit to
    # - https://jessicastringham.net/2017/12/31/stride-tricks/
    # - https://zhuanlan.zhihu.com/p/64933417
    assert len(src.shape) == 2, 'src must be a 2D array'
    assert kernel_size % 2 == 1, 'kernel size must be odd'
    src_pad = np.pad(src, kernel_size // 2, 'edge')
    shape = (
        src.shape[0],
        src.shape[1],
        kernel_size,
        kernel_size
    )
    strides = (
        src_pad.strides[0],
        src_pad.strides[1],
        src_pad.strides[0],
        src_pad.strides[1],
    )
    src_pad_extended = as_strided(src_pad, shape=shape, strides=strides)
    dst = np.min(src_pad_extended, axis=(2, 3))
    assert dst.shape == src.shape
    return dst


def dilation(src: np.array, kernel_size: int):
    '''dilation operation.
    .. math::
        dst(x,y) = \max_{(x', y') : kernel(x', y') \\neq 0} src(x + x', y + y')
    '''
    assert len(src.shape) == 2, 'src must be a 2D array'
    assert kernel_size % 2 == 1, 'kernel size must be odd'
    src_pad = np.pad(src, kernel_size // 2, 'edge')
    shape = (
        src.shape[0],
        src.shape[1],
        kernel_size,
        kernel_size
    )
    strides = (
        src_pad.strides[0],
        src_pad.strides[1],
        src_pad.strides[0],
        src_pad.strides[1],
    )
    src_pad_extended = as_strided(src_pad, shape=shape, strides=strides)
    dst = np.max(src_pad_extended, axis=(2, 3))
    assert dst.shape == src.shape
    return dst


def opening(img: np.array, kernel_size: int):
    """ opening operation
    """
    img_erode = erosion(img, kernel_size)
    img_dilate = dilation(img, kernel_size)
    return img_dilate


def closing(img: np.array, kernel_size: int):
    '''
    closing operation
    '''
    img_dilate = dilation(img, kernel_size)
    img_erode = erosion(img_dilate, kernel_size)
    return img_erode


def morph_grad(img: np.array, kernel_size: int):
    '''
    morphological gradient operation
    '''
    img_erose = erosion(img, kernel_size)
    img_dilute = dilation(img, kernel_size)
    return np.abs(img_erose.astype(np.int32) - img_dilute.astype(np.int32)).astype(np.uint8)


def top_hat(img: np.array, kernel_size: int):
    """ top hat operation
    """
    img_open = opening(img, kernel_size)
    return np.abs(img.astype(np.int32) - img_open.astype(np.int32)).astype(np.uint8)


def black_hat(img: np.array, kernel_size: int):
    """ black hat operation
    """
    img_close = closing(img, kernel_size)
    return np.abs(img_close.astype(np.int32) - img.astype(np.int32)).astype(np.uint8)


def binary(img: np.array, threshold: float):
    '''
    thresholding operation
    '''
    return (img > threshold) * 255


def convolve(src: np.array, kernel: np.array, pad_mode='edge'):
    assert len(src.shape) == 2, 'src must be a 2D array'
    assert len(kernel.shape) == 2, 'kernel must be a 2D array'
    assert kernel.shape[0] % 2 == 1, 'kernel size must be odd'
    assert kernel.shape[1] % 2 == 1, 'kernel size must be odd'
    src_pad = np.pad(src, ((kernel.shape[0] // 2, kernel.shape[0] // 2),
                           (kernel.shape[1] // 2, kernel.shape[1] // 2)), pad_mode)
    shape = (
        src.shape[0],
        src.shape[1],
        kernel.shape[0],
        kernel.shape[1]
    )
    strides = (
        src_pad.strides[0],
        src_pad.strides[1],
        src_pad.strides[0],
        src_pad.strides[1],
    )
    src_pad_extended = as_strided(src_pad, shape=shape, strides=strides)
    dst = np.tensordot(src_pad_extended, kernel, axes=([2, 3], [0, 1]))
    assert dst.shape == src.shape
    return dst


def sobel(img: np.array):
    '''
    sobel edge detection
    '''
    assert len(img.shape) == 2, 'src must be a 2D array'
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    img_x = np.abs(convolve(img, kernel_x))
    img_y = np.abs(convolve(img, kernel_y))
    img_edge = np.sqrt(img_x ** 2 + img_y ** 2)
    return img_edge, img_x, img_y


def get_gaussian_kernel(kernel_size: int, sigma: float):
    '''
    get gaussian kernel
    '''
    assert kernel_size % 2 == 1, 'kernel size must be odd'
    x = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    y = x[:, np.newaxis]
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


def gaussian_blur(src: np.array, kernel_size: int, sigma: float):
    '''
    gaussian blur
    '''
    assert len(src.shape) == 2, 'src must be a 2D array'
    assert kernel_size % 2 == 1, 'kernel size must be odd'
    kernel = get_gaussian_kernel(kernel_size, sigma)
    return convolve(src, kernel)


def reverse_binary(img: np.array):
    '''
    reverse binary operation
    '''
    return (img < 255) * 255


def laplace_transform(img: np.array):
    '''
    laplace transform
    '''
    assert len(img.shape) == 2, 'src must be a 2D array'
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return convolve(img, kernel)


def hough_transform(img: np.array, edge_img: np.array, theta_num: int, rho_num: int, top_k: int):
    '''
    hough transform
    '''
    assert len(img.shape) == 2, 'img must be a 2D array'
    assert img.shape == edge_img.shape, 'img and edge_img must be same shape'
    height, width = edge_img.shape[:2]
    height_half, width_half = height / 2, width / 2
    # initlize the hough space
    d = np.sqrt(height ** 2 + width ** 2)
    thetas = np.linspace(0, np.pi, theta_num)
    rhos = np.linspace(-d, d, rho_num)
    dtheta, drho = thetas[1] - thetas[0], rhos[1] - rhos[0]
    thetas_sep = np.append((thetas - dtheta / 2), (thetas[-1] + dtheta / 2))
    rhos_sep = np.append((rhos - dtheta / 2), (rhos[-1] + drho / 2))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    # extract edge points
    points = np.argwhere(edge_img > 0) - np.array([height_half, width_half])
    # rho = x \cos theta + y \in theta
    # rho_values : [N, num_thetas]
    rho_values = np.matmul(points, np.array([cos_thetas, sin_thetas]))
    # accumulate the votes
    # accumulator : [num_thetas, num_rhos]
    accumulator, _, _ = np.histogram2d(
        x=np.tile(thetas, reps=rho_values.shape[0]),  # [num_thetas * N]
        y=rho_values.ravel(),  # -> [num_rhos * N]
        bins=[thetas_sep, rhos_sep]
    )
    lines, = np.dstack(np.unravel_index(
        np.argsort(accumulator.ravel())[-top_k:], accumulator.shape))
    theta_idx, rho_idx = lines[:, 0], lines[:, 1]
    t, r = thetas[theta_idx], rhos[rho_idx]
    return t, r


def plot_img_line(img: np.array, thetas: np.array, rhos: np.array):
    '''
    plot image with lines. The axis origin is at the center of the image.
    '''
    assert len(thetas) == len(rhos), 'thetas and rhos must be same length'
    height, width = img.shape[:2]
    height_half, width_half = height / 2, width / 2
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    fig = plt.figure()
    subplot = fig.add_subplot(1, 1, 1)
    # ! To show the image correctly, we stipulate:
    # - The x axis of is vertical (from top to bottom),
    # - and the y axis of is horizontal (from left to right).
    subplot.set_xlim(0, width)
    subplot.set_ylim(0, height)
    subplot.set_ylabel("x")
    subplot.set_xlabel("y")
    subplot.imshow(img, cmap='gray')
    for cos_theta, sin_theta, rho in zip(cos_thetas, sin_thetas, rhos):
        if sin_theta != 0:
            # y = - \cos theta / \sin theta * x + rho / \sin theta
            x1, y1 = 1, -cos_theta / sin_theta * 1 + rho / sin_theta
            x0, y0 = 0, rho / sin_theta
        else:
            # x = rho / \cos theta
            x0, y0 = rho / cos_theta, 0
            x1, y1 = rho / cos_theta, 1
        x0, y0 = x0 + height_half, y0 + width_half
        x1, y1 = x1 + height_half, y1 + width_half
        # ! swap x and y coodrinates.
        subplot.axline(xy1=(y0, x0), xy2=(y1, x1), color='r')
    subplot.invert_yaxis()
    plt.show()
