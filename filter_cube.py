#!/usr/bin/env python -u

# Filter out emission from a THOR HI cube. Will 2D Fourier transform each plane of the cube and zero out the centre
# of the Fourier image and then inverse Fouroer transform back to the image domain. This produces a cube without the
# large scale emission

# Author James Dempsey
# Date 26 Nov 2017

from __future__ import print_function, division

import argparse
import sys
import time

from astropy.io import fits
import numpy as np
import pyfftw


def parseargs():
    """
    Parse the command line arguments
    :return: An args map with the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Filter the large scale emission from an imag cube using Fourier transforms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("input", help="The name of the file to be filtered.")
    parser.add_argument("output", help="The name of the filtered file to be produced.")
    parser.add_argument("-r", "--radius", help="The radius of the filter to apply to the centre of the Fourier image.",
                        default=20, type=int)
    parser.add_argument("-t", "--threads", help="The number of threads to be used for the Fourier transform.",
                        default=4, type=int)

    args = parser.parse_args()
    return args


def do_fftw(image, threads=2):
    """
    Calculate the Fourier transform of the input 2 dimensional image using the
    pyFFTW library.

    :param image: The square float64 image to be transformed.
    :param threads: The number of threads to be used by pyFFTW.
    :return: The fourier transform.
    """
    image_in = pyfftw.empty_aligned(image.shape, dtype='float64')
    image_in[:] = image
    fft_object = pyfftw.builders.fft2(image_in, axes=(0, 1), threads=threads)
    image_out = fft_object()
    return image_out


def do_ifftw(image, threads=2):
    """
    Calculate the inverse Fourier transform of the input 2 dimensional Fourier image using the
    pyFFTW library.

    :param image: The square complex128 image to be transformed.
    :param threads: The number of threads to be used by pyFFTW.
    :return: The fourier transform.
    """
    image_in = pyfftw.empty_aligned(image.shape, dtype='complex128')
    image_in[:] = image
    fft_object = pyfftw.builders.ifft2(image_in, axes=(0, 1), threads=threads)
    image_out = fft_object()
    return image_out


def fft_image(image, threads=4):
    """
    Produce a processed Fourier transform of the input image. The image must be
    square and of type float64 and real only. The Fourier transform will be
    shifted to have the zero-frequency component in the centre of the image.

    :param image: The square image to be transformed.
    :param threads: The number of threads to be used by pyFFTW.
    :return: The centred complex Fourier transform.
    """
    #ft_img = np.fft.fft2(image)
    ft_img = do_fftw(image, threads)
    #print(ft_img.shape)
    ft_shift = np.fft.fftshift(ft_img)

    return ft_shift


def ifft_image(ft_shift, threads=4):
    """
    Invert a Fourier transform of an image. The resulting image will be
    square and of type complex128. The real aspect of this image will represent the image.
    The Fourier transform will be unshifted to move the zero-frequency component away from the centre of the image.

    :param ft_shift: The centred complex Fourier transform.
    :param threads: The number of threads to be used by pyFFTW.
    :return: The complex inverse Fourier transformed image.
    """
    unshifted = np.fft.ifftshift(ft_shift)
    #inverted = np.fft.ifft2(unshifted)
    inverted = do_ifftw(unshifted, threads=threads)

    return inverted


def filter_plane(plane, radius=20, threads=4):
    # Prepare the spatial slice for fft
    start = time.time()
    flipped = np.concatenate((plane, np.fliplr(plane)), axis=1)
    mirrored = np.concatenate((flipped, np.flipud(flipped)), axis=0)
    x_pad = (mirrored.shape[0] - mirrored.shape[1]) // 2
    padded = np.lib.pad(mirrored, ((0, 0), (x_pad, x_pad)), 'constant')
    prep_end = time.time()
    print('   Prep for plane took %.02f s' % (prep_end - start))
    sys.stdout.flush()

    # Do the fft
    ft_img = fft_image(padded, threads)
    ft_end = time.time()
    print('   FFT for plane took %.02f s' % (ft_end - prep_end))
    sys.stdout.flush()

    # Filter out the large scsle emission
    centre_y = ft_img.shape[0] // 2
    centre_x = ft_img.shape[1] // 2
    ft_img[centre_y - radius:centre_y + radius, centre_x - radius:centre_x + radius] = 0

    # Invert the fft to get back the image
    inverted = ifft_image(ft_img, threads)
    ift_end = time.time()
    print('   iFFT for plane took %.02f s' % (ift_end - ft_end))
    sys.stdout.flush()
    post_psd_2d = inverted.real

    centre_y = post_psd_2d.shape[0] // 2
    centre_x = post_psd_2d.shape[1] // 2
    post_plane = post_psd_2d[:centre_y, x_pad:centre_x].astype(np.float32)

    return post_plane


def filter_image(image, radius=40, threads=4):
    #pyfftw.interfaces.cache.enable()
    filtered = np.zeros(image.shape, dtype=np.float32)

    for idx in range(image.shape[0]):
        print("Processing plane", idx)
        sys.stdout.flush()

        plane = image[idx, :, :]
        post_plane = filter_plane(plane, radius, threads)
        filtered[idx, :, :] = post_plane

    return filtered


def load_image(filename):
    hdulist = fits.open(filename, memmap=True)
    image = hdulist[0].data
    print("Image shape is", image.shape)
    header = hdulist[0].header
    return image, header


def save_image(filename, image, header, radius):
    header['history'] = "Emission filtered with radius {} Fourier filter.".format(radius)
    hdu = fits.PrimaryHDU(image, header)
    hdu.writeto(filename, overwrite=True)


def main():
    """
    Main script for filter_cube
    :return: The exit code
    """
    args = parseargs()

    start = time.time()

    print("#### Started filtering of cube {} at {} ####".format(args.input,
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))))

    # Filter the image
    orig_image, header = load_image(args.input)

    filtered = filter_image(orig_image, radius=args.radius, threads=args.threads)

    save_image(args.output, filtered, header, args.radius)

    # Report
    end = time.time()
    print('#### Filtering completed at %s ####' %
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end)))
    print('Filtering took %.02f s' %
          (end - start))
    return 0


# Run the script if it is called from the command line
if __name__ == "__main__":
    exit(main())
