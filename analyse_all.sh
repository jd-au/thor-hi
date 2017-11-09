#!/usr/bin/env bash

# Script to run the absorption analysis of a series of THOR HI cubes.
# 
# The HI cubes are expected to be in a folder called hidata and a source 
# catalogue named catalog.dat is expected to be present in the current folder.
#
# First absorption spectra will be extracted from all cubes
# Then the spectra will be analysed in bulk, grouped by rating and 
# plots and a catalogue will be output

startday=`date +%Y%m%d`

mkdir -p logs

date
let COUNTER=1
for filename in hidata/*.fits; do
    echo "Processing $filename"
    python analyse_data.py ${filename} >& logs/analyse-${COUNTER}.log

    let COUNTER=COUNTER+1
done

echo "Analysing spectra"
python analyse_spectra.py >& logs/spectra.log

date
echo "Done"