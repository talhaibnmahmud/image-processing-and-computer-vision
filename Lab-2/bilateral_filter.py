"""
This script implements a bilateral filter on an image withou using OpenCV.
"""

import concurrent.futures
import os
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt


class BilateralFilter:
    """
    Bilaterally filters an image.
    Applies a filter to the image which is made up of a Gaussian filter with standard deviation sigma_d and
    a range filter with standard deviation sigma_r.
    """

    def __init__(self, filter_size: int, sigma_d: float, sigma_r: float):
        self.filter_size = filter_size
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r

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

    def get_blocks(self, img: cv2.Mat, padding: int):
        """
        Returns a list of tuples containing the start and end indices of the rows to be processed by the multiprocessing
        function.

        :param img: The image to be processed.
        :param padding: The padding to be applied to the image.
        :return: List of row indices.
        """
        width = img.shape[0]

        processors = os.cpu_count()
        num_processes = processors if processors is not None else 8
        block_size = width // num_processes

        blocks: list[tuple[int, int]] = []
        for i in range(num_processes):
            start = i * block_size
            end = start + block_size + padding * 2

            if width - end + padding * 2 < block_size:
                end += width - end + padding * 2

            blocks.append((start, end))

        return blocks

    def apply_filter(
        self,
        img: cv2.Mat,
        kernel: npt.NDArray[np.float32],
        kernel_size: int,
        start: int,
        end: int
    ) -> tuple[npt.NDArray[np.int16], int, int]:
        """
        Applies the bilateral filter to the image.

        :param img: The image to be processed.
        :param kernel: The Gaussian filter.
        :param kernel_size: The size of the Gaussian filter.
        :param start: The start index of the rows to be processed.
        :param end: The end index of the rows to be processed.
        :return: The filtered image and the start and end indices of the rows to be processed.
        """
        _row, col = img.shape[:2]
        padding = kernel_size // 2

        padded_img = cv2.copyMakeBorder(                    # type: ignore
            img,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_REPLICATE
        )

        # output_dimension = (end - start - 2 * padding + 1, col)
        result = np.zeros(img.shape[:2], dtype=np.int16)    # type: ignore
        sigma_r = self.sigma_r
        s_r_squared = 2 * sigma_r ** 2
        range_filter = np.zeros(                            # type: ignore
            (kernel_size, kernel_size),
            dtype=np.float32
        )

        # print(f'Processing block {start} to {end}')
        # print(f'Output dimension: {output_dimension}')
        # print(f'Looping over {start + padding, end - padding} rows')

        for x in range(start + padding, end - padding):
            for y in range(padding, col + padding):
                value = 0.0

                center = int(padded_img[x, y])
                for i in range(-padding, padding + 1):
                    for j in range(-padding, padding + 1):
                        # Intensity difference between the pixel and the kernel center
                        diff = int(padded_img[x + i][y + j]) - center
                        range_filter[i - padding][j - padding] = np.exp(
                            -diff**2 / s_r_squared) / s_r_squared

                filter = np.multiply(kernel, range_filter)
                weight = np.sum(filter)                     # type: ignore

                for i in range(-padding, padding + 1):
                    for j in range(-padding, padding + 1):
                        value += padded_img[x + i, y + j] * \
                            filter[i + padding, j + padding]
                result[x - padding, y - padding] = value // weight
                # print(f'Pixel ({x - padding}, {y - padding}) processed')

        # print(f'Block {start} to {end - padding * 2} processed')
        return result, start, end - padding * 2

    @staticmethod
    def arg_wrapper(args: Any) -> tuple[npt.NDArray[np.int16], int, int]:
        """
        Wrapper for the apply_filter function.
        It is necessary to use this wrapper because the apply_filter function receives multiple arguments.
        But ProcessPoolExecutor.map() only accepts a single argument.

        :param args: The arguments to be passed to the apply_filter function.
        :return: The filtered image and the start and end indices of the rows to be processed.
        """
        return BilateralFilter.apply_filter(*args)

    def apply_multiprocessing(
        self, img: cv2.Mat,
        filter: npt.NDArray[np.float32],
        filter_size: int,
        blocks: list[tuple[int, int]]
    ) -> npt.NDArray[np.int16]:
        """
        Applies the bilateral filter to the image using multiprocessing.

        :param img: The image to be processed.
        :param filter: The Gaussian filter.
        :param filter_size: The size of the Gaussian filter.
        :param blocks: The list of row indices to be processed.
        :return: The filtered image.
        """
        args = [(self, img, filter, filter_size, *item) for item in blocks]
        output_img = np.zeros(img.shape, dtype=np.int16)    # type: ignore

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(BilateralFilter.arg_wrapper, args)

            for result in results:
                output_img[result[1]:result[2],
                           :] = result[0][result[1]:result[2], :]

        return output_img

    def apply(self, img: cv2.Mat, parallel: bool = True) -> npt.NDArray[np.int16]:
        """
        Applies the bilateral filter to the image.

        :param img: The image to be processed.
        :param parallel: Whether to use multiprocessing.
        :return: The filtered image.
        """
        kernel = self.get_gaussian_filter(self.filter_size, self.sigma_d)
        blocks = self.get_blocks(img, self.filter_size // 2)

        if parallel:
            output_img = self.apply_multiprocessing(
                img, kernel, self.filter_size, blocks)
        else:
            start, end = 0, img.shape[0] + (self.filter_size // 2) * 2
            output_img = self.apply_filter(
                img, kernel, self.filter_size, start, end)[0]
        return output_img


if __name__ == '__main__':
    img: cv2.Mat = cv2.imread('./images/rubiks_cube.png', 0)
    bi_filter = BilateralFilter(5, 1, 5)
    output_img = bi_filter.apply(img)

    cv2.imshow('Input', img)            # type: ignore
    cv2.imshow('Output', output_img)    # type: ignore
    cv2.waitKey(0)
