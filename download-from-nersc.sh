#!/bin/bash

lines=("OII_DOUBLET" "HGAMMA" "HBETA" "OIII_4959" "OIII_5007" "NII_6548" "HALPHA" "NII_6584" "SII_6716" "SII_6731")
#lines=("OII_DOUBLET")
for l in ${lines[*]}; do
    for i in {0..10}; do
        scp  -i ~/.ssh/nersc ashodkh@perlmutter-p1.nersc.gov:/pscratch/sd/a/ashodkh/fluxes_from_spectra/raw_masked/sv3_fluxes_raw_masked${i}"_selection5_"${l}"_bins13.txt.npz" ~/Desktop/research/git_repos/emission-lines/data/fluxes/
    done
done

