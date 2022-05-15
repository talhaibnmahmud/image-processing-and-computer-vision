from statistics import median
import cv2
import numpy as np
import numpy.typing as npt


class Filters:
    def __init__(self, img: cv2.Mat):
        self.img = img

    def convolve(self, kernel: npt.NDArray[np.float32], ksize: int) -> npt.NDArray[np.int16]:
        row, col = self.img.shape[:2]
        padding = ksize // 2

        padded_img = cv2.copyMakeBorder(
            self.img, padding, padding, padding, padding, cv2.BORDER_REPLICATE
        )
        output = np.zeros((row, col), np.int16)

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
        _kernel = np.ones((ksize, ksize), np.float32)
        padding = ksize // 2
        row, col = self.img.shape[:2]
        padded_img = cv2.copyMakeBorder(
            self.img, padding, padding, padding, padding, cv2.BORDER_REPLICATE
        )
        output = np.zeros((row, col), np.uint8)

        for x in range(padding, row + padding):
            for y in range(padding, col + padding):
                pixels: list[np.uint8] = []

                for i in range(-padding, padding + 1):
                    for j in range(-padding, padding + 1):
                        pixels.append(padded_img[x + i, y + j])

                output[x - padding, y -
                       padding] = np.sum(pixels) // len(pixels)
        return output

    def median(self, wsize: int = 3) -> npt.NDArray[np.uint8]:
        """
        Applies median filter to the image.
        :param ksize: size of the kernel
        :return: image with median filter applied
        """
        row, col = self.img.shape[:2]
        padding = wsize // 2

        padded_img = cv2.copyMakeBorder(
            self.img, padding, padding, padding, padding, cv2.BORDER_REPLICATE
        )
        output = np.zeros((row, col), np.uint8)

        for x in range(padding, row + padding):
            for y in range(padding, col + padding):
                pixels: list[np.uint8] = []

                for i in range(-padding, padding + 1):
                    for j in range(-padding, padding + 1):
                        pixels.append(padded_img[x + i, y + j])

                pixels.sort()
                output[x - padding, y - padding] = pixels[len(pixels) // 2]

        return output

    def get_gaussian_filter(self, filter_size: int, sigma: float) -> npt.NDArray[np.float32]:
        """
        Returns a 2D Gaussian filter of size filter_size x filter_size and standard deviation sigma.

        :return: The Gaussian filter.
        """
        spread = filter_size // 2
        x, y = np.mgrid[-spread:spread +
                        1, -spread:spread + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / g.sum()

    def gaussian(self, ksize: int = 3, sigma: float = 0.5) -> npt.NDArray[np.uint8]:
        """
        Applies gaussian filter to the image.
        :param ksize: size of the kernel
        :param sigma: standard deviation of the gaussian distribution
        :return: image with gaussian filter applied
        """
        kernel = self.get_gaussian_filter(ksize, sigma)
        return self.convolve(kernel, ksize)

        # return cv2.GaussianBlur(self.img, (ksize, ksize), sigma)

    def laplacian(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
        """
        Applies laplacian filter to the image.
        :param ksize: size of the kernel
        :return: image with laplacian filter applied
        """
        p_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
        n_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)

        return self.convolve(p_kernel, ksize), self.convolve(n_kernel, ksize)

    def _horizontal_sobel_kernel(self, ksize: int):
        row = [[0 for _ in range(ksize)] for _ in range(1)]
        col = [[1 for _ in range(1)] for _ in range(ksize)]
        center = ksize // 2
        col[center][0] = 2

        for i in range(1):
            for j in range(ksize):
                row[i][j] = center
                center -= 1

        return np.matmul(col, row)

    def _vertical_sobel_kernel(self, kernel_size: int):
        col = [[0 for _ in range(1)] for _ in range(kernel_size)]
        row = [[1 for _ in range(kernel_size)] for _ in range(1)]
        center = kernel_size // 2
        row[0][center] = 2

        for i in range(kernel_size):
            for j in range(1):
                col[i][j] = center
                center -= 1

        return np.matmul(col, row)

    def sobel(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], ...]:
        """
        Applies sobel filter to the image.
        :param ksize: size of the kernel
        :return: image with sobel filter applied
        """
        kernel_x = self._horizontal_sobel_kernel(ksize)
        kernel_y = self._vertical_sobel_kernel(ksize)

        horizontal = self.convolve(kernel_x, ksize)
        vertical = self.convolve(kernel_y, ksize)

        result = np.zeros(self.img.shape[:2], np.int16)

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                result[i][j] = np.sqrt(
                    horizontal[i][j] ** 2 + vertical[i][j] ** 2
                )

        return horizontal, vertical, result

    def _horizontal_prewitt_kernel(self, ksize: int = 3):
        return np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    def _vertical_prewitt_kernel(self, ksize: int = 3):
        return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    def prewitt(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], ...]:
        """
        Applies prewitt filter to the image.
        :param ksize: size of the kernel
        :return: image with prewitt filter applied
        """
        kernel_x = self._horizontal_prewitt_kernel(ksize)
        kernel_y = self._vertical_prewitt_kernel(ksize)

        horizontal = self.convolve(kernel_x, ksize)
        vertical = self.convolve(kernel_y, ksize)

        result = np.zeros(self.img.shape[:2], np.int16)

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                result[i][j] = np.sqrt(
                    horizontal[i][j] ** 2 + vertical[i][j] ** 2
                )

        return horizontal, vertical, result

    def _horizontal_roberts_kernel(self, ksize: int = 2):
        return np.array([[1, 0], [0, -1]])

    def _vertical_roberts_kernel(self, ksize: int = 2):
        return np.array([[0, 1], [-1, 0]])

    def roberts(self, ksize: int = 3) -> cv2.Mat:
        """
        Applies roberts filter to the image.
        :param ksize: size of the kernel
        :return: image with roberts filter applied
        """
        kernel_x = self._horizontal_prewitt_kernel(ksize)
        kernel_y = self._vertical_prewitt_kernel(ksize)

        horizontal = self.convolve(kernel_x, ksize)
        vertical = self.convolve(kernel_y, ksize)

        result = np.zeros(self.img.shape[:2], np.int16)

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                result[i][j] = np.sqrt(
                    horizontal[i][j] ** 2 + vertical[i][j] ** 2
                )

        return horizontal, vertical, result

    def _horizontal_scharr_kernel(self):
        return np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])

    def _vertical_scharr_kernel(self):
        return np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])

    def scharr(self, ksize: int = 3) -> tuple[npt.NDArray[np.int16], ...]:
        """
        Applies scharr filter to the image.
        :param ksize: size of the kernel
        :return: image with scharr filter applied
        """
        kernel_x = self._horizontal_scharr_kernel()
        kernel_y = self._vertical_scharr_kernel()

        horizontal = self.convolve(kernel_x, ksize)
        vertical = self.convolve(kernel_y, ksize)

        result = np.zeros(self.img.shape[:2], np.int16)

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                result[i][j] = np.sqrt(
                    horizontal[i][j] ** 2 + vertical[i][j] ** 2
                )

        return horizontal, vertical, result
