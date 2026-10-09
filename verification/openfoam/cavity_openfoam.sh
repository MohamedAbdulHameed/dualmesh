#!/bin/bash
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Lid-driven cavity with OpenFOAM's icoFoam at Re = 100, sampled along the two
# centrelines so that the profiles can be compared with dualmesh and with the
# reference data of Ghia, Ghia and Shin (1982).
#
# Usage:  ./cavity_openfoam.sh [resolution]
# Needs:  an OpenFOAM installation; set FOAM_BASHRC if it is not in the
#         default location.

if [ -z "$WM_PROJECT_DIR" ]; then
  # shellcheck disable=SC1090
  . "${FOAM_BASHRC:-/usr/lib/openfoam/openfoam2406/etc/bashrc}"
fi
if [ -z "$WM_PROJECT_DIR" ]; then
  echo "OpenFOAM was not found; set FOAM_BASHRC to its etc/bashrc" >&2
  exit 1
fi

RESOLUTION=${1:-64}
HERE=$(cd "$(dirname "$0")" && pwd)
CASE="$HERE/run/cavity"

rm -rf "$CASE"
mkdir -p "$CASE"
cp -r "$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity"/* "$CASE"/ || exit 1
cd "$CASE" || exit 1

# The tutorial is L = 0.1 m, U = 1 m/s, nu = 0.01 m^2/s, i.e. Re = 10.
# Set nu = 0.001 m^2/s for Re = U L / nu = 100.
sed -i 's/^nu .*/nu              0.001;/' constant/transportProperties
sed -i "s/^\( *hex (0 1 2 3 4 5 6 7) \)(20 20 1)/\1($RESOLUTION $RESOLUTION 1)/" system/blockMeshDict
sed -i 's/^endTime .*/endTime         20;/' system/controlDict
sed -i 's/^deltaT .*/deltaT          0.0005;/' system/controlDict
sed -i 's/^writeInterval .*/writeInterval   4000;/' system/controlDict

blockMesh > log.blockMesh 2>&1 || { echo "blockMesh failed, see $CASE/log.blockMesh"; exit 1; }
icoFoam > log.icoFoam 2>&1 || { echo "icoFoam failed, see $CASE/log.icoFoam"; exit 1; }

cat > system/sampleDict <<'SAMPLE'
FoamFile { version 2.0; format ascii; class dictionary; object sampleDict; }
type sets;
libs ("libsampling.so");
interpolationScheme cellPoint;
setFormat raw;
fields (U);
sets
(
    vertical
    {
        type    uniform;
        axis    y;
        start   (0.05 0.0   0.005);
        end     (0.05 0.1   0.005);
        nPoints 129;
    }
    horizontal
    {
        type    uniform;
        axis    x;
        start   (0.0  0.05  0.005);
        end     (0.1  0.05  0.005);
        nPoints 129;
    }
);
SAMPLE
postProcess -func sampleDict -latestTime > log.sample 2>&1 \
  || { echo "sampling failed, see $CASE/log.sample"; exit 1; }
echo "OpenFOAM results are in $CASE/postProcessing/sampleDict"
