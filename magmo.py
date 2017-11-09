# Set of general use functions for processing MAGMO HI data.
#

# Author James Dempsey
# Date 29 Jul 2016

import csv
import sys
import os
import subprocess


class CommandFailedError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# functions to be made common
def get_metadata_file_path(filename):
    """
    Take a metadata file name and get the path to it given that it will be in
    the same folder as this script.

    :param filename: The filename of the metadata file.
    :return: The absolute path of the file (including the filename)
    """
    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, filename)
    return abs_file_path


def get_day_file_data(day):
    """
    Read the magmo-days-full.csv file and find the row for the requested day.
    :param day: The day to be found
    :return: The day's row, or None if the day is not defined
    """

    with open(get_metadata_file_path('magmo-days-full.csv'), 'rb') as magmodays:
        reader = csv.reader(magmodays)
        for row in reader:
            if row[0] == day:
                return row
    return None


def get_day_obs_data(day):
    """
    Read the magmo-obs.csv file and find the rows for the requested day.
    :param day: The day to be found
    :return: The day's rows, or None if the day is not defined
    """

    sources = []
    with open(get_metadata_file_path('magmo-obs.csv'), 'rb') as magmo_obs:
        reader = csv.reader(magmo_obs)
        for row in reader:
            if row[0] == day:
                src = dict()
                src['source'] = row[4]
                src['phase_cal'] = row[10]
                src['gal_l'] = row[2]
                sources.append(src)
    return sources


def run_os_cmd(cmd, failOnErr=True):
    """
    Run an operating system command ensuring that it finishes successfully.
    If the comand fails, the program will exit.
    :param cmd: The command to be run
    :return: None
    """
    print ">", cmd
    sys.stdout.flush()
    try:
        retcode = subprocess.call(cmd, shell=True)
        if retcode != 0:
            message = "Command '"+cmd+"' failed with code " + str(retcode)
            print >>sys.stderr, message
            if failOnErr:
                raise CommandFailedError(message)
    except OSError as e:
        message = "Command '" + cmd + "' failed " + e
        print >> sys.stderr, message
        if failOnErr:
            raise CommandFailedError(message)
    return None


def ensure_dir_exists(dirname):
    """
    Check if a folder exists, and if it doesn't, create it. Fail the
    program if the folder could not be created.
    :param dirname: The name of the folder to be created.
    :return: None
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not os.path.isdir(dirname):
        print "Directory %s could not be created." % dirname
        exit(1)
    return None


def get_continuum_ranges():
    """
    Read in the predefined velocity ranges for different longitudes over which
    the continuum will be measured. Each range will have keys min_long,
    max_long, min_con_vel, max_con_vel. All values will be integers,

    :return: A list of dictionaries.
    """

    continuum_ranges = []
    with open(get_metadata_file_path('magmo-continuum.csv'), 'rb') as con_def:
        reader = csv.reader(con_def)
        first = True
        for row in reader:
            if first:
                first = False
            else:
                continuum_ranges.append(
                    {'min_long': int(row[0]), 'max_long': int(row[1]),
                     'min_con_vel': int(row[2]), 'max_con_vel': int(row[3])})

    return continuum_ranges


def lookup_continuum_range(continuum_ranges, longitude):
    """
    Lookup the velocity range that shouild be used for measuring the continuum
    levels at a particular longitude.

    :param continuum_ranges: The list of continuum ranges.
    :param longitude: The integer longitude to be checked.
    :return: The min and max continuum velocities.
    """
    if longitude < 0:
        longitude += 360
    continuum_start_vel = -210
    continuum_end_vel = -150
    for row in continuum_ranges:
        if row['min_long'] <= longitude <= row['max_long']:
            continuum_start_vel = row['min_con_vel']
            continuum_end_vel = row['max_con_vel']
    return continuum_start_vel, continuum_end_vel
