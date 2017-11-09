#!/usr/bin/env python -u

# Analyse the produced HI spectra and extract stats and produce diagrams.
#
# This program reads the previously generated spectra and stats files in the day
# folders and produces initial stage analysis data products. These include
# histograms of spectra produced vs used, and longitude velocity diagrams.

# Author James Dempsey
# Date 29 Sep 2016

from __future__ import print_function, division

import argparse
import csv
import datetime
import glob
import os
import re
import time

from astropy.coordinates import SkyCoord, matching
from astropy.io import fits, votable
from astropy.io.votable import parse, from_table, writeto
from astropy.table import Table, Column
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from scipy import ndimage

import magmo


class Field(object):
    """
    "name", "rms", "max", "sn", "strong"
    """

    def __init__(self, day, name, rms, max_flux, sn_ratio, used, longitude,
                 latitude):
        self.day = day
        self.name = name
        self.rms = rms
        self.max_flux = max_flux
        self.sn_ratio = sn_ratio
        self.used = used

        self.longitude = longitude
        self.latitude = latitude

    def get_field_id(self):
        return str(self.day) + "-" + str(self.name)


class Spectrum(object):
    """

    """

    def __init__(self, day, field_name, src_id, longitude, latitude, velocity,
                 opacities, flux):
        self.day = day
        self.field_name = field_name
        self.src_id = src_id
        self.longitude = longitude
        self.latitude = latitude
        self.velocity = velocity
        self.opacities = opacities
        self.flux = flux
        self.low_sn = None
        self.range = 0
        self.opacity_range = 0
        self.max_s_max_n = 0
        self.continuum_sd = 0
        self.rating = 'A'

    def get_field_id(self):
        return str(self.day) + "-" + str(self.field_name)

    def __str__(self):
        return self.get_field_id() + ", src: " + str(self.src_id)


def parseargs():
    """
    Parse the command line arguments
    :return: An args map with the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Analyse the produced HI spectra and extract stats and produce diagrams.")
    # parser.add_argument("-i", "--input", help="The input spectra catalogue",
    #                    default='magmo-spectra.vot')
    # parser.add_argument("--plot_only", help="Produce plots for the result of a previous decomposition", default=False,
    #                    action='store_true')

    args = parser.parse_args()
    return args


def read_spectra():
    """
    Read in the spectra produced in earlier pipeline stages.

    :return: An array of Spectrum objects
    """
    spectra = []

    vo_files = glob.glob('spectra/*/*_opacity.votable.xml')
    print("Reading {} spectrum files.".format(len(vo_files)))
    for filename in sorted(vo_files):
        print ('Reading', filename)
        votable = parse(filename, pedantic=False)
        results = next(resource for resource in votable.resources if
                       resource.type == "results")
        if results is not None:
            gal_long = None
            gal_lat = None
            for info in votable.infos:
                if info.name == 'longitude':
                    gal_long = float(info.value)
                    if gal_long > 180:
                        gal_long -= 360
                if info.name == 'latitude':
                    gal_lat = float(info.value)
                if info.name == 'beam_area':
                    beam_area = float(info.value)
                if info.name == 'gname':
                    gname = info.value
            if gal_long is None:
                print("No longitude provided for %s, skipping" % filename)
                continue
            results_array = results.tables[0].array

            velocities = np.zeros(len(results_array))
            opacities = np.zeros(len(results_array))
            fluxes = np.zeros(len(results_array))
            em_temps = np.zeros(len(results_array))
            em_std = np.zeros(len(results_array))
            sigma_tau = np.zeros(len(results_array))
            i = 0
            has_emission = False
            for row in results_array:
                opacity = row['opacity']
                velocities[i] = row['velocity'] / 1000.0
                opacities[i] = opacity
                fluxes[i] = row['flux']
                if 'sigma_tau' in results_array.dtype.names:
                    sigma_tau[i] = row['sigma_tau']
                if 'em_mean' in results_array.dtype.names:
                    em_temps[i] = row['em_mean']
                    has_emission = True
                if 'em_std' in results_array.dtype.names:
                    em_std[i] = row['em_std']
                i += 1
            paths = filename.split('/')
            field = paths[-1].split('_')
            spectrum = Spectrum('0', field[0], field[1][3:],
                                gal_long, gal_lat, velocities, opacities,
                                fluxes)
            spectrum.temp_brightness = results_array['temp_brightness']
            min_opacity = np.min(spectrum.opacities)
            max_opacity = np.max(spectrum.opacities)

            continuum_ranges = magmo.get_continuum_ranges()

            opacity_range = max_opacity - min_opacity
            max_s_max_n = (1 - min_opacity) / (max_opacity - 1)
            continuum_sd, continuum_temp = calc_continuum_sd(spectrum, continuum_ranges)
            rating = calc_rating(opacity_range, max_s_max_n, continuum_sd, em_std)

            loc = SkyCoord(gal_long, gal_lat, frame='galactic', unit="deg")
            spectrum.loc = loc
            spectrum.ra = loc.icrs.ra.degree
            spectrum.dec = loc.icrs.dec.degree
            spectrum.name = name_spectrum(gname)

            spectrum.opacity_range = opacity_range
            spectrum.max_s_max_n = max_s_max_n
            spectrum.continuum_sd = continuum_sd
            spectrum.continuum_temp = continuum_temp
            spectrum.rating = rating
            spectrum.beam_area = beam_area
            spectrum.em_temps = em_temps
            spectrum.em_std = em_std
            spectrum.sigma_tau = sigma_tau
            spectrum.has_emission = has_emission
            spectra.append(spectrum)

    return spectra


def name_spectrum(id):
    return id
    #precision = 1000
    #glong = (loc.galactic.l.degree * precision // 1) / precision
    #glat = (loc.galactic.b.degree * precision // 1) / precision
    #return 'MAGMOHI G{:0=7.3f}{:=+06.3f}'.format(glong, glat)


def read_field_stats():
    """
    Read in all of the stats csv files for the days processed and build a list
    of fields.

    :return: List of fields.
    """

    fields = []
    field = Field(0, 'unknown', 0, 0, 2, True,
                                  0, 0)
    fields.append(field)

    stats_files = glob.glob('day*/stats.csv')
    print("Reading {} day stats files.".format(len(stats_files)))
    for filename in sorted(stats_files):
        # print ('Reading', filename)
        day = int(filename.split('/')[0][3:])

        with open(filename, 'rb') as stats:
            reader = csv.reader(stats)
            first = True
            for row in reader:
                if first:
                    first = False
                else:
                    if "-" in row[0]:
                        coords = row[0].split('-')
                        coords[1] = "-" + coords[1]
                    else:
                        coords = row[0].split('+')

                    used = 'N'
                    if len(row) > 4:
                        used = row[4]
                    field = Field(day, row[0], row[1], row[2], row[3], used,
                                  float(coords[0]), float(coords[1]))
                    fields.append(field)

    return fields


def extract_lv(spectra, min_rating='C'):
    x = []
    y = []
    c = []
    bad_spectra = 0
    duplicate_spectra = 0
    prev_field = ''
    num_fields = 0
    used_fields = []

    for spectrum in spectra:
        opacities = spectrum.opacities
        if spectrum.rating > min_rating:
            bad_spectra += 1
            spectrum.low_sn = True
            continue
        if spectrum.duplicate:
            duplicate_spectra += 1
            continue
        y = np.concatenate((y, spectrum.velocity))
        c = np.concatenate((c, opacities))
        x = np.concatenate((x, np.full(len(opacities), spectrum.longitude)))
        field_id = spectrum.get_field_id()
        if field_id != prev_field:
            prev_field = field_id
            used_fields.append(field_id)
            num_fields += 1

    print("In %d fields read %d spectra of which %d had reasonable S/N and %d were duplicates, leaving %d to plot. " % (
        num_fields, len(spectra), len(spectra) - bad_spectra, duplicate_spectra,
        len(spectra) - bad_spectra - duplicate_spectra))

    return x, y, c, used_fields


def plot_lv(x, y, c, filename, continuum_ranges, zoom):
    """
    Produce a longitude-velocity diagram from the supplied data and write it
    out to a file.

    :param x: The x-coordinate of each point (galactic longitude in degrees)
    :param y: The y-coordinate of each point (LSR velocity in km/s)
    :param c: The opacity fraction (1= transparent, 0=completely opaque)
    :param filename: The file to write the plot to.
    :param continuum_ranges: The file to write the plot to.
    :param zoom: Should the plot be zoomed in on the data region
    ,
    :return: None
    """
    xmin = 5 if zoom else -180
    xmax = 90 if zoom else 180
    ymin = -300
    ymax = 300

    # print("X: %d, Y: %d, data: %d" % (len(x), len(y), len(c) ))
    val = np.clip(c, -0.005, 1.05)

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[1] = 4.5
    plt.rcParams["figure.figsize"] = fig_size

    # print val
    fig = plt.figure(1, (12, 6))
    # plt.subplots_adjust(hspace=0.5)
    plt.subplot(111, axisbg='black' if zoom else 'gray')
    plt.hexbin(x, y, val, cmap=plt.cm.gist_heat_r)
    # plt.scatter(x, y, cmap=plt.cm.YlOrRd_r)
    plt.axis([xmax, xmin, ymin, ymax])
    plt.title("Longitude-Velocity")
    plt.xlabel('Galactic longitude (deg)')
    plt.ylabel('LSR Velocity (km/s)')
    cb = plt.colorbar(orientation='horizontal')
    #cb = plt.colorbar()
    cb.set_label(r'$e^{(-\tau)}$')

    # Add bands for the continuum ranges
    if not zoom:
        for con_range in continuum_ranges:
            min_l = con_range['min_long']
            max_l = con_range['max_long']
            if min_l < 181:
                min_x = 0.5 - (min_l / 360.0)
                max_x = 0.5 - (max_l / 360.0)
            else:
                min_x = 0.5 + ((360 - min_l) / 360.0)
                max_x = 0.5 + ((360 - max_l) / 360.0)
            plt.axhline(con_range['min_con_vel'], xmin=min_x,
                        xmax=max_x, color='blue')
            # linestyle='dashed')
            plt.axhline(con_range['max_con_vel'], xmin=min_x,
                        xmax=max_x, color='blue')
            # linestyle='dashed')

    plt.grid(color='White')

    plt.savefig(filename)
    plt.close()
    print("Plotted ", len(c), "opacity points to", filename)

    return


def world_to_pixel(header, axis, value):
    """
    Calculate the pixel value for the provided world value using the WCS
    keywords on the specific axis. The axis must be linear.
    :param header: The FITS header describing the zxes.
    :param axis:  The number of the target axis.
    :param value: The world value to be converted.
    :return: The pixel value.
    """
    ax = str(axis)
    return int(header['CRPIX' + ax] + (value - header['CRVAL' + ax]) / header[
        'CDELT' + ax])


def get_lv_subset(data, header, l_min, l_max, v_min, v_max):
    """
    Extract a subset of velocity, longitude data based on physical bounds.

    :param data: The two dimensional array of data.
    :param header: The FITS header of the data with axes of longitude, velocity.
    :param l_min: The minimum of the desired longitude range.
    :param l_max: The maximum of the desired longitude range.
    :param v_min: The minimum of the desired velocity range.
    :param v_max: The maximum of the desired velocity range.
    :return: A numpy array with only the data from the requested range.
    """

    l_start = world_to_pixel(header, 1, l_max)
    l_end = world_to_pixel(header, 1, l_min)
    v_start = world_to_pixel(header, 2, v_min)
    v_end = world_to_pixel(header, 2, v_max)

    return data[v_start:v_end, l_start:l_end]


def plot_lv_image(x, y, c, filename):
    """
    Output a longitude-velocity plot of the provided data with the outline of
    emission from the GASS III dataset plotted over the data.

    :param x: The longitude value of each data point.
    :param y: The velocity value of each data point.
    :param c: The optical depth value of each data point.
    :param filename: The file name of the plot.
    :return: None
    """
    # Image dimensions
    l_max = 90
    l_min = 5
    l_dpd = 1 / 0.08
    l_size = int((l_max - l_min) * l_dpd)
    v_max = 300
    v_min = -300
    v_dpkms = 1 / 0.8245
    v_size = int((v_max - v_min) * v_dpkms)

    val = np.clip(c, -0.005, 1.05)

    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'

    dots_per_degree = l_dpd  # 4*3
    data = ma.array(np.ones((v_size, l_size)), mask=True)
    # print(data)
    xmax = data.shape[1]
    ymax = data.shape[0]
    for i in range(0, len(x)):
        x_idx = xmax - int((x[i] - l_min) * dots_per_degree)
        y_idx = ymax - int((y[i] - v_min) * v_dpkms)
        data[y_idx, x_idx - 3:x_idx + 4] = val[i]

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[1] = 4.5
    plt.rcParams["figure.figsize"] = fig_size

    smoothed_data = ndimage.gaussian_filter(data, sigma=2, order=0)

    ax = plt.subplot(111)
    img = ax.imshow(smoothed_data, cmap=plt.cm.gist_heat_r)
    plt.title("Longitude-Velocity")
    plt.xlabel('Galactic longitude (deg)')
    plt.ylabel('LSR Velocity (km/s)')

    #cbaxes = plt.add_axes([0.05, 0.05, 0.9, 0.025])
    #cb = plt.colorbar(cax=cbaxes, mappable=mappable, orientation='horizontal')
    cb = plt.colorbar(img, ax=ax, orientation='horizontal')

    #cb = plt.colorbar(img, ax=ax)
    cb.set_label(r'$e^{(-\tau)}$')

    gass_lv = fits.open('gass-lv.fits')
    gass_subset = get_lv_subset(gass_lv[0].data, gass_lv[0].header, l_min,
                                l_max, v_min * 1000,
                                v_max * 1000)

    # Add an outline of the emission from GASS III
    #contour_set = plt.contour(np.log10(np.flipud(gass_subset)), 1, cmap='Pastel2')
    #plt.clabel(contour_set)

    # Set the axis ticks and scales
    x_step = int(20 * (gass_subset.shape[1] / (l_max - l_min)))
    ax.set_xticks([i for i in range(gass_subset.shape[1], 0, -x_step)])
    ax.set_xticklabels([i for i in range(l_min, l_max + 1, 20)])

    y_step = int(100 * (gass_subset.shape[0] / (v_max - v_min)))
    ax.set_yticks([i for i in range(0, gass_subset.shape[0], y_step)])
    ax.set_yticklabels([i for i in range(v_max, v_min - 1, -100)])

    plt.grid(color='antiquewhite')

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def calc_continuum_sd(spectrum, continuum_ranges):
    """
    Calulate the standard deviaition of opacity in the velocity range
    designated as continuum for the spectrum's longitude. This gives a measure
    of the noise in wat should be an otherwise continuum only part of the
    spectrum.

    :param spectrum: The spectrum object being analysed.
    :param continuum_ranges: The defined contionuum ranges
    :return: The opacity standard deviation and average temperature
    """

    continuum_start_vel, continuum_end_vel = magmo.lookup_continuum_range(
        continuum_ranges, int(spectrum.longitude))
    continuum_range = np.where(
        continuum_start_vel < spectrum.velocity)
    bin_start = continuum_range[0][0]
    continuum_range = np.where(
        spectrum.velocity < continuum_end_vel)
    bin_end = continuum_range[0][-1]
    sd_cont = np.std(spectrum.opacities[bin_start:bin_end])
    cont_temp = np.mean(spectrum.temp_brightness)
    return sd_cont, cont_temp


def calc_rating(opacity_range, max_s_max_n, continuum_sd, em_std):
    rating_codes = 'ABCDEF'
    rating = 0

    if opacity_range > 1.5:
        rating += 1
    if max_s_max_n < 3:
        rating += 1
    if continuum_sd*3 > 1:
        rating += 1

    return rating_codes[rating]


def is_resolved(day, field_name, island_id, source_id, beam_area, islands):
    """
    Identify if this spectrum is for a resolved source. A sourceis resolved if its area is larger than the beam area.

    :param day: The day of the observation
    :param field_name: The field observed
    :param island_id: The id of the source island
    :param source_id: The component id within the island
    :param beam_area: The area of the beam in steradians
    :return: True if the source is resolved, False otherwise.
    """
    for isle in islands:
        #print (isle['Day'], day, isle['Field'], field_name, isle['island'], island_id)
        if isle['Day'] == day and isle['Field'] == field_name and isle['island'] == int(island_id):
            #src_area = math.radians(src['a']/3600.0) * math.radians(src['b']/3600.0)
            print ("island %s %s %s is %f as compared to beam of %f" % (day, field_name, island_id,  isle['area'], isle['beam_area']))
            return isle['area'] > isle['beam_area']


def set_field_metadata(field, ucd, unit, description):
    if ucd:
        field.ucd = ucd
    if unit:
        field.unit = unit
    if description:
        field.description = description


def output_spectra_catalogue(spectra):
    """
    Output the list of spectrum stats to a VOTable file thor-hi-spectra.vot

    :param spectra: The list of Spectrum objects
    :return: None
    """
    rows = len(spectra)
    ids = np.empty(rows, dtype=object)
    days = np.zeros(rows, dtype=int)
    fields = np.empty(rows, dtype=object)
    sources = np.empty(rows, dtype=object)
    longitudes = np.zeros(rows)
    latitudes = np.zeros(rows)
    eq_ras = np.zeros(rows)
    eq_decs = np.zeros(rows)
    max_flux = np.zeros(rows)
    max_opacity = np.zeros(rows)
    min_opacity = np.zeros(rows)
    max_velocity = np.zeros(rows)
    min_velocity = np.zeros(rows)
    rms_opacity = np.zeros(rows)
    opacity_range = np.zeros(rows)
    continuum_sd = np.zeros(rows)
    continuum_temp = np.zeros(rows)
    max_s_max_n = np.zeros(rows)
    max_em_std = np.zeros(rows)
    max_emission = np.zeros(rows)
    min_emission = np.zeros(rows)
    rating = np.empty(rows, dtype=object)
    used = np.empty(rows, dtype=bool)
    resolved = np.empty(rows, dtype=bool)
    duplicate = np.empty(rows, dtype=bool)
    has_emission = np.empty(rows, dtype=bool)
    filenames = np.empty(rows, dtype=object)
    local_paths = np.empty(rows, dtype=object)
    local_emission_paths = np.empty(rows, dtype=object)

    base_path = os.path.realpath('.')
    i = 0
    for spectrum in spectra:
        ids[i] = spectrum.name
        days[i] = int(spectrum.day)
        fields[i] = spectrum.field_name
        sources[i] = spectrum.src_id
        longitudes[i] = spectrum.longitude
        latitudes[i] = spectrum.latitude
        eq_ras[i] = spectrum.ra
        eq_decs[i] = spectrum.dec
        max_flux[i] = np.max(spectrum.flux)
        min_opacity[i] = np.min(spectrum.opacities)
        max_opacity[i] = np.max(spectrum.opacities)
        rms_opacity[i] = np.sqrt(np.mean(np.square(spectrum.opacities)))
        min_velocity[i] = np.min(spectrum.velocity)
        max_velocity[i] = np.max(spectrum.velocity)
        max_em_std[i] = np.max(spectrum.em_std)
        if spectrum.has_emission:
            max_emission[i] = np.max(spectrum.em_temps)
            min_emission[i] = np.min(spectrum.em_temps)

        opacity_range[i] = spectrum.opacity_range
        max_s_max_n[i] = spectrum.max_s_max_n
        continuum_sd[i] = spectrum.continuum_sd
        continuum_temp[i] = spectrum.continuum_temp
        rating[i] = spectrum.rating
        src_parts = spectrum.src_id.split('-')
        resolved[i] = False # is_resolved(spectrum.day, spectrum.field_name, src_parts[0], src_parts[1], spectrum.beam_area,
                            #      None)

        duplicate[i] = spectrum.duplicate
        has_emission[i] = spectrum.has_emission
        used[i] = not spectrum.low_sn
        prefix = 'spectra/' + spectrum.field_name + '/' + spectrum.field_name + "_src" + spectrum.src_id
        filenames[i] =  prefix + "_plot.png"
        em_filename = prefix + "_emission.png"
        local_paths[i] = base_path + '/' + filenames[i]
        local_emission_paths[i] = base_path + '/' + em_filename
        i += 1

    spectra_table = Table(
        [ids, days, fields, sources, longitudes, latitudes, eq_ras, eq_decs, max_flux, min_opacity,
         max_opacity, rms_opacity, min_emission, max_emission, min_velocity, max_velocity, used, continuum_temp,
         opacity_range, max_s_max_n, continuum_sd, max_em_std, rating, resolved, duplicate, has_emission,
         filenames, local_paths, local_emission_paths],
        names=['Name', 'Day', 'Field', 'Source', 'Longitude', 'Latitude', 'RA', 'Dec', 'Max_Flux',
               'Min_Opacity', 'Max_Opacity', 'RMS_Opacity', 'Min_Emission', 'Max_Emission', 'Min_Velocity',
               'Max_Velocity', 'Used', 'Continuum_Temp', 'Opacity_Range', 'Max_S_Max_N',
               'Continuum_SD', 'max_em_std', 'Rating', 'Resolved', 'Duplicate', 'Has_Emission',
               'Filename', 'Local_Path', 'Local_Emission_Path'],
        meta={'ID': 'thor_hi_spectra',
              'name': 'THOR HI Spectra ' + str(datetime.date.today())})
    votable = from_table(spectra_table)
    table = votable.get_first_table()
    set_field_metadata(table.get_field_by_id('Min_Emission'), 'stat.min', 'K',
                       'Minimum average emission')
    set_field_metadata(table.get_field_by_id('Max_Emission'), 'stat.max', 'K',
                       'Maximum average emission')

    filename = "thor-hi-spectra.vot"
    writeto(votable, filename)
    print("Wrote out", i, "spectra to", filename)
    for grade in "ABCDEF":
        num_rating = len(np.where(rating == grade)[0])
        print ("%s: %3d" % (grade, num_rating))
    print ("Mean continuum sd %f" % np.mean(continuum_sd))


def flag_duplicate_fields(fields):
    """
    Examine the list of fields and identify those fields that are duplicate observations and should not be used. Where
    a field was observed more than once, the observation with the highest signal to noise ratio will be the only one
    used. A new duplicate value will be added to each field, with a value of true for those which should not be used.

    :param fields: The list of fields
    :return: The map of fields against their day and name.
    """
    unique_field_map = dict()
    full_field_map = dict()
    for field in fields:
        full_field_map[field.get_field_id()] = field
        if field.name in unique_field_map:
            prev_field = unique_field_map.get(field.name)
            if field.sn_ratio > prev_field.sn_ratio:
                prev_field.duplicate = True
                print("Marking day %s field %s as duplicate (sn %.03f < %.03f for day %s)" % (
                    prev_field.day, prev_field.name, float(prev_field.sn_ratio), float(field.sn_ratio), field.day))
            else:
                field.duplicate = True
                print("Marking day %s field %s as duplicate (sn %.03f < %.03f for day %s)" % (
                    field.day, field.name, float(field.sn_ratio), float(prev_field.sn_ratio), prev_field.day))
                continue
        field.duplicate = False
        unique_field_map[field.name] = field
    print("Marked %d fields as duplicates out of %d" % (len(fields) - len(unique_field_map), len(fields)))
    return full_field_map


def output_field_catalogue(fields, used_fields):
    """
    Write out a catalogue of the fields observed under the MAGMO project
    with some basic stats for each field.

    :param fields: The fields to be written.
    :param used_fields: An aray of field ids which had spectra which were used.
    :return: None
    """
    rows = len(fields)
    days = np.zeros(rows, dtype=int)
    field_names = np.empty(rows, dtype=object)
    longitudes = np.zeros(rows)
    latitudes = np.zeros(rows)
    max_fluxes = np.zeros(rows)
    sn_ratios = np.zeros(rows)
    strong = np.empty(rows, dtype=bool)
    used = np.empty(rows, dtype=bool)
    duplicate = np.empty(rows, dtype=bool)

    i = 0
    for field in fields:
        days[i] = int(field.day)
        field_names[i] = field.name
        longitudes[i] = field.longitude
        latitudes[i] = field.latitude
        max_fluxes[i] = field.max_flux
        sn_ratios[i] = field.sn_ratio
        sn_ratios[i] = field.sn_ratio
        strong[i] = True if field.used == 'Y' else False
        used[i] = field.get_field_id() in used_fields
        duplicate[i] = field.duplicate
        i += 1

    coords = SkyCoord(longitudes, latitudes, frame='galactic', unit="deg")

    fields_table = Table(
        [days, field_names, longitudes, latitudes, max_fluxes, sn_ratios,
         strong, used, duplicate, coords.icrs.ra.degree, coords.icrs.dec.degree],
        names=['Day', 'Field', 'Longitude',
               'Latitude', 'Max_Flux', 'SN_Ratio', 'Strong', 'Used', 'Duplicate', 'ra', 'dec'],
        meta={'ID': 'thor_fields',
              'name': 'THOR Fields ' + str(datetime.date.today())})
    votable = from_table(fields_table)
    filename = "thor-fields.vot"
    writeto(votable, filename)

    print("Wrote out", i, "fields to", filename)


def read_sources(filename, sources):
    print ("Extracting sources from " + filename)

    src_votable = votable.parse(filename, pedantic=False)

    # Add day and field info
    pattern = re.compile('day([0-9]+)/([0-9.+-]+)_src_[a-z]*.vot')
    result = pattern.match(filename)
    day = result.group(1)
    field = result.group(2)
    num_rows = len(src_votable.get_first_table().array)
    day_data = np.repeat([day], num_rows)
    field_data = np.repeat([field], num_rows)
    src_table = src_votable.get_first_table().to_table()
    src_table.add_column(Column(name='Day', data=day_data), index=0)
    src_table.add_column(Column(name='Field', data=field_data), index=1)

    if sources is None:
        sources = src_table
    else:
        for row in src_table:
            sources.add_row(row)

    return sources


def output_source_catalogue():
    vo_files = glob.glob('day*/*_src_comp.vot')
    sources = None
    for vof in vo_files:
        sources = read_sources(vof, sources)

    # Write out the catalogue
    sources.meta['name'] = 'THOR Sources ' + str(datetime.date.today())
    vot = votable.from_table(sources)
    vot.to_xml("thor-sources.vot")

    vo_files = glob.glob('day*/*_src_isle.vot')
    islands = None
    for vof in vo_files:
        islands = read_sources(vof, islands)

    # Write out the catalogue
    islands.meta['name'] = 'THOR Islands ' + str(datetime.date.today())
    vot = votable.from_table(islands)
    vot.to_xml("thor-islands.vot")

    # Create a map by day of the islands
    isle_day_map = {}
    for isle in islands:
        day = isle['Day']
        if day not in isle_day_map:
            isle_day_map[day] = []
        day_list = isle_day_map[day]
        day_list.append(isle)

    return isle_day_map


def output_single_phase_catalogue(spectra):
    """
    Create a catalogue of the spin temperature of each channel of each spectrum based on a naive single phase
    assumption.
    :param spectra: The list of all spectra
    :return: None
    """
    spectra_by_long = sorted(spectra, key=lambda spectrum: spectrum.longitude)

    longitudes = []
    latitudes = []
    velocities = []
    emission_temps = []
    opacities = []
    spin_temperatures = []

    for spectrum in spectra_by_long:
        for i in range(len(spectrum.velocity)):
            if spectrum.em_temps[i] > 0:
                longitudes.append(spectrum.longitude)
                latitudes.append(spectrum.latitude)
                velocities.append(spectrum.velocity[i])
                emission_bright_temp = spectrum.em_temps[i]
                emission_temps.append(emission_bright_temp)
                opacities.append(spectrum.opacities[i])
                spin_t = None
                if emission_bright_temp:
                    spin_t = emission_bright_temp / spectrum.opacities[i]
                spin_temperatures.append(spin_t)

    temp_table = Table(
        [longitudes, latitudes, velocities, spin_temperatures, emission_temps, opacities],
        names=['Longitude', 'Latitude', 'Velocity', 'Spin_Temp', 'Emission_Bright_Temp', 'Opacity'],
        meta={'ID': 'thor_single_phase_spin_temp',
              'name': 'THOR 1P Spin Temp ' + str(datetime.date.today())})
    votable = from_table(temp_table)
    filename = "thor-1p-temp.vot"
    writeto(votable, filename)

    print("Wrote out", len(spin_temperatures), "channel temperatures to", filename)


def plot_spectra(spectra):
    magmo.ensure_dir_exists("plots")
    for rating in 'ABCDEF':
        magmo.ensure_dir_exists("plots/" + rating)

    continuum_ranges = magmo.get_continuum_ranges()

    for spectrum in spectra:
        if spectrum.em_temps is None or len(spectrum.em_temps) == 0:
            # skip entries which have no emission data
            continue

        if spectrum.duplicate:
            # Skip duplicates
            continue

        con_start_vel, con_end_vel = magmo.lookup_continuum_range(
            continuum_ranges, int(spectrum.longitude))
        print ("longitude of {} gave range of {} - {}".format(int(spectrum.longitude), con_start_vel, con_end_vel))

        # Plot line chart of bright_temp vs opacity according in velocity order
        fig = plt.figure(0, [6, 7])

        # 1. emission
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(spectrum.velocity, spectrum.em_temps)
        em_max = spectrum.em_temps + spectrum.em_std
        em_min = spectrum.em_temps - spectrum.em_std
        plt.fill_between(spectrum.velocity, em_min, em_max, facecolor='lightgray', color='lightgray')
        ax.axvline(con_start_vel, color='g', linestyle='dashed')
        ax.axvline(con_end_vel, color='g', linestyle='dashed')

        #ax.axhline(0, color='r')
        #ax.set_xlabel(r'Velocity relative to LSR (km/s)')
        ax.set_ylabel(r'$T_B (K)$')
        ax.grid(True)
        ax.set_title(spectrum.name + " (" + spectrum.rating + ")\n" + spectrum.field_name + " src " + spectrum.src_id +
                     " on day " + spectrum.day)
        plt.setp(ax.get_xticklabels(), visible=False)

        # 2. absorption
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
        ax2.plot(spectrum.velocity, spectrum.opacities)
        ax2.axhline(1, color='r')

        if len(spectrum.sigma_tau) > 0:
            tau_max = 1 + spectrum.sigma_tau
            tau_min = 1 - spectrum.sigma_tau
            ax2.fill_between(spectrum.velocity, tau_min, tau_max, facecolor='lightgray', color='lightgray')

        ax2.set_xlabel(r'Velocity relative to LSR (km/s)')
        ax2.set_ylabel(r'$e^{(-\tau)}$')
        ax2.grid(True)
        ax2.axvline(con_start_vel, color='g', linestyle='dashed')
        ax2.axvline(con_end_vel, color='g', linestyle='dashed')

        # 3. scatter
        #ax3 = fig.add_subplot(3, 1, 3)
        #ax3.plot(1-spectrum.opacities, spectrum.em_temps, markersize=2, marker='o')
        #ax3.set_xlabel(r'$1 - e^{(-\tau)}$')
        #ax3.set_ylabel(r'$T_B (K)$')
        #ax3.grid(True)

        plt.tight_layout()
        # change axis location of ax5
        #pos1 = ax.get_position()
        #pos2 = ax2.get_position()
        #points1 = pos1.get_points()
        #points2 = pos2.get_points()
        #points2[1][1] = points1[0][1]
        #pos2.set_points(points2)
        #ax2.set_position(pos2)

        # Write out to plots/field-day-src.pdf
        filename = spectrum.name + ".pdf"
        plt.savefig("plots/" + spectrum.rating + "/" + filename)
        plt.close()


def filter_duplicate_sources(spectra, field_map):
    duplicate_radius = 4 * u.arcsec
    print ('## Identifying duplicate sources (reobserved fields or within {} arcsec'.format(duplicate_radius))

    spectra_array = np.array(spectra)
    l = []
    b = []
    duplicate_spectra = 0
    for spectrum in spectra_array:
        l.append(spectrum.loc.galactic.l.degree)
        b.append(spectrum.loc.galactic.b.degree)
        # Flag spectra from duplicate fields
        spectrum.duplicate = False
    num_spectra_dupe_fields = duplicate_spectra

    thor_coords = SkyCoord(l, b, frame='galactic', unit="deg")

    idx_match1, idx_match2, dist_2d, dist_3d = thor_coords.search_around_sky(thor_coords, duplicate_radius)
    for i in range(len(idx_match1)):
        match1 = spectra_array[idx_match1[i]]
        match2 = spectra_array[idx_match2[i]]
        if not match1.duplicate and not match2.duplicate and match1 != match2:
            #print("{} is only {:.2f} arcsec from {} Rating {} v {} ContSD {:.3f} v {:.3f}".format(match1.name,
            #                                                                                  dist_2d[i].to(u.arcsec),
            #                                                                          match2.name, match1.rating,
            #                                                                          match2.rating,
            #                                                                          match1.continuum_sd,
            #                                                                          match2.continuum_sd))
            if match1.rating > match2.rating or (not match1.has_emission and match2.has_emission) or \
                            match1.continuum_sd > match2.continuum_sd:
                match1.duplicate = True
                print("Flagging {} rating {} as duplicate. Dist {:.2f}".format(match1.name, match1.rating,
                                                                           dist_2d[i].to(u.arcsec)))
            else:
                match2.duplicate = True
                print("Flagging {} rating {} as duplicate. Dist {:.2f}".format(match2.name, match2.rating,
                                                                           dist_2d[i].to(u.arcsec)))
            duplicate_spectra += 1
    print(
        "Flagged {} spectra out of {} as duplicates. {} based on field, {} based on distance".format(duplicate_spectra,
                                                                                                     len(spectra),
                                                                                                     num_spectra_dupe_fields,
                                                                                                     duplicate_spectra - num_spectra_dupe_fields))


def main():
    """
    Main script for analyse_spectra
    :return: The exit code
    """
    args = parseargs()

    start = time.time()

    print("#### Started analysis of THOR HI spectra at %s ####" %
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start)))

    # Read Fields
    fields = read_field_stats()
    field_map = flag_duplicate_fields(fields)

    # Output source catalogue
    #isle_day_map = output_source_catalogue()

    # Process Spectra
    spectra = read_spectra()
    filter_duplicate_sources(spectra, field_map)
    x, y, c, used_fields = extract_lv(spectra)
    continuum_ranges = magmo.get_continuum_ranges()
    plot_lv(x, y, c, 'thor-lv.pdf', continuum_ranges, False)
    plot_lv(x, y, c, 'thor-lv-zoom.pdf', continuum_ranges, True)
    plot_lv_image(x, y, c, 'thor-lv-zoom-im.pdf')
    output_spectra_catalogue(spectra)


    # calculate single phase spin temp for A-C
    #output_single_phase_catalogue(spectra)

    plot_spectra(spectra)

    # Output only the really good spectra
    x, y, c, temp = extract_lv(spectra, min_rating='B')
    plot_lv(x, y, c, 'thor-lv_AB.pdf', continuum_ranges, False)
    plot_lv_image(x, y, c, 'thor-lv-zoom-im_AB.pdf')

    # Process Fields
    output_field_catalogue(fields, used_fields)

    # also want
    # - Catalogue - Fields - day, field, peak, sn, coords, used
    # - Catalogue - Source - field, source, continuum, min, max, sn, used
    # - Histogram - fields observed/used per day
    # - Histogram - fields observed/used per day

    # Report
    end = time.time()
    print('#### Analysis completed at %s ####' %
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end)))
    print('Processed spectra in %.02f s' %
          (end - start))
    return 0


# Run the script if it is called from the command line
if __name__ == "__main__":
    exit(main())
