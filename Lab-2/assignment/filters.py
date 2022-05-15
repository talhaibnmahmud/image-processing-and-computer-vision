"""
This script implements various filters on an image withou using OpenCV.
"""

import cv2
import numpy as np
import numpy.typing as npt


class Filters:
    """ Filter Class """

    def __init__(self, img: cv2.Mat):
        """
        :param img: image to be processed
        """
        self.img = img

    def convolve(self, kernel: npt.NDArray[np.float32], ksize: int) -> npt.NDArray[np.int16]:
        """
        Applies convolution to the image.
        :param kernel: kernel to be applied
        :param ksize: size of the kernel
        :return: image with convolution applied
        """
        row, col = self.img.shape[:2]
        padding = ksize // 2

        padded_img = cv2.copyMakeBorder(                    # type: ignore
            self.img, padding, padding, padding, padding, cv2.BORDER_REPLICATE
        )
        output = np.zeros((row, col), np.int16)             # type: ignore

        for x in range(padding, row + padding):
            for y in range(padding, col + padding):
                value = 0.0

                for i in range(-padding, padding + 1):
                    for j in range(-padding, padding + 1):
                        value += padded_img[x + i, y + j] * \
                            kernel[i + padding, j + padding]

                output[x - padding, y - padding] = value

        return output

    def mean(self, ksize: int = 3) -> npt.NDArray[np.uint8]:
        """
        Applies mean filter to the image.
        :param ksize: size of the kernel
        :return: image with mean filter applied
        """
        padding = ksize // 2
        row, col = self.img.shape[:2]
        padded_img = cv2.copyMakeBorder(                    # type: ignore
            self.img, padding, padding, padding, padding, cv2.BORDER_REPLICATE
        )
        output = np.zeros((row, col), np.uint8)             # type: ignore

        for x in range(padding, row + padding):
            for y in range(padding, col + padding):
                pixels: list[np.uint8] = []

                for i in range(-padding, padding + 1):
                    for j in range(-padding, padding + 1):
                        pixels.append(padded_img[x + i, y + j])

                output[x - padding, y -
                       padding] = np.sum(pixels) // len(pixels)  # type: ignore
        return output

    def median(self, wsize: int = 3) -> npt.NDArray[np.uint8]:
        """
        Applies median filter to the image.
        :param ksize: size of the kernel
        :return: image with median filter applied
        """
        row, col = self.img.shape[:2]
        padding = wsize // 2

        padded_img = cv2.copyMakeBorder(                    # type: ignore
            self.img, padding, padding, padding, padding, cv2.BORDER_REPLICATE
        )
        output = np.zeros((row, col), np.uint8)             # type: ignore

        for x in range(padding, row + padding):
            for y in range(padding, col + padding):
                pixels: list[np.uint8] = []

                for i in range(-padding, padding + 1):
                    for j in range(-padding, padding + 1):
                        pixels.append(padded_img[x + i, y + j])

                pixels.sort()                               # type: ignore
                output[x - padding, y - padding] = pixels[len(pixels) // 2]

        return output

    def get_gaussian_filter(self, filter_size: int, sigma: float) -> npt.NDArray[np.float32]:
        """
        Returns a 2D Gaussian filter of size filter_size x filter_size and standard deviation sigma.
        :param filter_size: size of the filter
        :param sigma: standard deviation of the gaussian distribution
        :return: The Gaussian filter.
        """
        spread = filter_size // 2
        x, y = np.mgrid[-spread:spread +
                        1, -spread:spread + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / g.sum()

    def gaussian(self, ksize: int = 3, sigma: float = 0.5) -> npt.NDArray[np.int16]:
        """
        Applies gaussian filter to the image.
        :param ksize: size of the kernel
        :param sigma: standard deviation of the gaussian distribution
        :return: image with gaussian filter applied
        """
        kernel = self.get_gaussian_filter(ksize, sigma)
        return self.convolve(kernel, ksize)

    def laplacian(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        """
        Applies laplacian filter to the image.
        :param ksize: size of the kernel
        :return: image with laplacian filter applied
        """
        p_kernel = np.array(                                # type: ignore
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            np.float32
        )
        n_kernel = np.array(                                # type: ignore
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            np.float32
        )

        p_img = self.convolve(p_kernel, ksize)
        n_img = self.convolve(n_kernel, ksize)

        p_min, p_max = p_img.min(), p_img.max()             # type: ignore
        n_min, n_max = n_img.min(), n_img.max()             # type: ignore

        for i in range(p_img.shape[0]):
            for j in range(p_img.shape[1]):
                p_img[i, j] = (p_img[i, j] - p_min) / (p_max - p_min) * 255
                n_img[i, j] = (n_img[i, j] - n_min) / (n_max - n_min) * 255

        return p_img, n_img

    def split_filter(
        self,
        ksize: int,
        kernel_x: npt.NDArray[np.float32],
        kernel_y: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.int16], ...]:
        """
        Applies horizontal and vertical filters to the image.
        :param ksize: size of the kernel
        :param kernel_x: horizontal kernel
        :param kernel_y: vertical kernel
        :return: image with horizontal and vertical filters applied
        """
        horizontal = self.convolve(kernel_x, ksize)
        vertical = self.convolve(kernel_y, ksize)

        result = np.zeros(self.img.shape[:2], np.int16)     # type: ignore

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                result[i][j] = np.sqrt(
                    horizontal[i][j] ** 2 + vertical[i][j] ** 2
                )

        return horizontal, vertical, result

    def _horizontal_sobel_kernel(self, ksize: int):
        """
        Returns the horizontal sobel kernel.
        :param ksize: size of the kernel
        :return: horizontal sobel kernel
        """
        row = [[0 for _ in range(ksize)] for _ in range(1)]
        col = [[1 for _ in range(1)] for _ in range(ksize)]
        center = ksize // 2
        col[center][0] = 2

        for i in range(1):
            for j in range(ksize):
                row[i][j] = center
                center -= 1

        return np.matmul(col, row)                          # type: ignore

    def _vertical_sobel_kernel(self, kernel_size: int):
        """
        Returns the vertical sobel kernel.
        :param kernel_size: size of the kernel
        :return: vertical sobel kernel
        """
        col = [[0 for _ in range(1)] for _ in range(kernel_size)]
        row = [[1 for _ in range(kernel_size)] for _ in range(1)]
        center = kernel_size // 2
        row[0][center] = 2

        for i in range(kernel_size):
            for j in range(1):
                col[i][j] = center
                center -= 1

        return np.matmul(col, row)                          # type: ignore

    def sobel(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], ...]:
        """
        Applies sobel filter to the image.
        :param ksize: size of the kernel
        :return: image with sobel filter applied
        """
        kernel_x = self._horizontal_sobel_kernel(ksize)
        kernel_y = self._vertical_sobel_kernel(ksize)

        return self.split_filter(ksize, kernel_x, kernel_y)

    def _horizontal_prewitt_kernel(self, ksize: int = 3):
        """
        Returns the horizontal prewitt kernel.
        :param ksize: size of the kernel
        :return: horizontal prewitt kernel
        """
        return np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])   # type: ignore

    def _vertical_prewitt_kernel(self, ksize: int = 3):
        """
        Returns the vertical prewitt kernel.
        :param ksize: size of the kernel
        :return: vertical prewitt kernel
        """
        return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])   # type: ignore

    def prewitt(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], ...]:
        """
        Applies prewitt filter to the image.
        :param ksize: size of the kernel
        :return: image with prewitt filter applied
        """
        kernel_x = self._horizontal_prewitt_kernel(ksize)
        kernel_y = self._vertical_prewitt_kernel(ksize)

        return self.split_filter(ksize, kernel_x, kernel_y)

    def _horizontal_roberts_kernel(self, ksize: int = 2):
        """
        Returns the horizontal roberts kernel.
        :param ksize: size of the kernel
        :return: horizontal roberts kernel
        """
        return np.array([[1, 0], [0, -1]])                  # type: ignore

    def _vertical_roberts_kernel(self, ksize: int = 2):
        """
        Returns the vertical roberts kernel.
        :param ksize: size of the kernel
        :return: vertical roberts kernel
        """
        return np.array([[0, 1], [-1, 0]])                  # type: ignore

    def roberts(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], ...]:
        """
        Applies roberts filter to the image.
        :param ksize: size of the kernel
        :return: image with roberts filter applied
        """
        kernel_x = self._horizontal_prewitt_kernel(ksize)
        kernel_y = self._vertical_prewitt_kernel(ksize)

        return self.split_filter(ksize, kernel_x, kernel_y)

    def _horizontal_scharr_kernel(self):
        """
        Returns the horizontal scharr kernel.
        :return: horizontal scharr kernel
        """
        return np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])  # type: ignore

    def _vertical_scharr_kernel(self):
        """
        Returns the vertical scharr kernel.
        :return: vertical scharr kernel
        """
        return np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])  # type: ignore

    def scharr(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], ...]:
        """
        Applies scharr filter to the image.
        :param ksize: size of the kernel
        :return: image with scharr filter applied
        """
        kernel_x = self._horizontal_scharr_kernel()
        kernel_y = self._vertical_scharr_kernel()

        return self.split_filter(ksize, kernel_x, kernel_y)
