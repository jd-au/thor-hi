# Set of general use functions for accessing SGPS HI data
#

# Author James Dempsey
# Date 07 Dec 2016

from __future__ import print_function, division

import os
import logging
import glob
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS


_SGPS_FOLDER = "/priv/myrtle1/gaskap/SGPS/"
_SGPS_HI_PATTERN = '*hi.fits.gz'


class Spectrum(object):
    """
    A comtainer for a spectrum
    """

    def __init__(self, coord, velocity, flux):
        self.coord = coord
        self.velocity = velocity
        self.flux = flux

    def __str__(self):
        return self.coord


def get_hi_file_list():
    """
    Retrieve a list of the HI SGPS data files and their longitude ranges.
    :return: A list of dictionaries.
    """
    hi_files = []
    sgps_path = get_sgps_location()
    if not os.path.exists(sgps_path):
        logging.warning("Unable to find SGPS files at " + sgps_path)
    fits_files = glob.glob1(sgps_path, _SGPS_HI_PATTERN)
    for fits_file in fits_files:
        fits_path = os.path.join(sgps_path, fits_file)
        sgps_fits = fits.open(fits_path, memmap=True)
        header = sgps_fits[0].header
        long_refpix = int(header['CRPIX1'])
        long_refval = float(header['CRVAL1'])
        long_delta = float(header['CDELT1'])
        long_naxis = int(header['NAXIS1'])
        del header
        sgps_fits.close()
        min_long = long_refval - (long_refpix-1) * long_delta
        max_long = long_refval + long_naxis * long_delta
        if long_delta < 0:
            temp = max_long
            max_long = min_long
            min_long = temp
        hi_files.append({'min_long': min_long, 'max_long': max_long,
                         'file_name': fits_path})
    return hi_files


def get_hi_file_at_long(longitude, sgps_hi_file_list, edge_size=0.5):
    """
    Identify the best SGPS file to retrieve data from at the supplied Galactic
    longitude. Will try to find a file where the longitude is at least edge_size
    away from the edge of the cube, but will fall back to any file with data
    covering that longtidue. Negative and positive longitudes for the third and
    fourth quadrants are automatically handled.
    :param longitude: The galactic longitude in decimal degrees, may be in the
                      range -180 < l < 360
    :param sgps_hi_file_list: A list of dictionaries describing the SGPS files.
    :param edge_size: The sie around the edge to, ideally, avoid.
    :return:
    """
    actual_long = longitude if longitude >= 0 else 360.0 + longitude
    for entry in sgps_hi_file_list:
        if entry['min_long']+edge_size < actual_long < entry['max_long']-edge_size:
            return entry['file_name']

    # Fallback to no buffer if there are no files with this value away
    # from the edges
    if edge_size > 0.0:
        return get_hi_file_at_long(actual_long, sgps_hi_file_list, edge_size=0.0)

    # We don't have coverage for this longitude
    return None


def extract_spectra(coords, sgps_hi_file_list):
    """
    Extract SGPS HI spectra at each of the galactic coordinates
    :param coords: The coordinates of the target locations as SkyCoord objects
    :param sgps_hi_file_list: A list of dictionaries describing the SGPS files.
    :return: A list of spectra objacts, order may not match the coords
    """

    # get files for the coords
    files = []
    file_indexes = []
    for coord in coords:
        hi_file = get_hi_file_at_long(coord.galactic.l.value, sgps_hi_file_list)
        if hi_file:
            if not (hi_file in files):
                files.append(hi_file)
            file_indexes.append(files.index(hi_file))

    results = []
    file_idx = 0
    for fits_filename in files:
        hdulist = fits.open(fits_filename)
        image = hdulist[0].data
        header = hdulist[0].header
        w = WCS(header)
        index = np.arange(header['NAXIS3'])
        velocities = w.wcs_pix2world(0, 0, index[:], 0, 0)[2]

        # For each target coord in this file
        for i in range(0,len(file_indexes)):
            if file_idx == file_indexes[i]:
                coord = coords[i]
                # convert desired coord to pixel coords
                pix = w.wcs_world2pix(coord.galactic.l, coord.galactic.b, 0, 0, 1)
                x_coord = int(round(pix[0])) - 1
                y_coord = int(round(pix[1])) - 1
                #print("Translated %.4f, %.4f to %d, %d" % (
                #    coord.galactic.l.value, coord.galactic.b.value, x_coord, y_coord))

                # Extract slice
                slice = image[0, :, y_coord, x_coord]

                # Create spectra object
                spectrum = Spectrum(coord, velocities, slice)

                # Add to result list
                results.append(spectrum)

        del image
        del header
        hdulist.close()
        file_idx += 1

    return results


def extract_image(coords, sgps_hi_file_list):
    # find the file which covers the coords
    # TODO: Which plane? total, sample where there is absorption in MAGMO?
    # Extract a cutout from the image
    # Plot the cutout
    # Plot a rectangle of the coordinates
    return


def set_sgps_location(sgps_folder):
    global _SGPS_FOLDER
    _SGPS_FOLDER = sgps_folder


def get_sgps_location():
    """
    Retreive the path to the SGPS data files. This may be set by the application
    or set in an environment variable SGPS_LOC
    :return: the path to the SGPS data files.
    """
    if os.environ.has_key('SGPS_LOC'):
        env_loc = os.environ['SGPS_LOC']
        if env_loc and os.path.isdir(env_loc):
            return env_loc
        else:
            logging.warning('Invalid SGPS_LOC value ignored: ' + env_loc)
    return _SGPS_FOLDER
