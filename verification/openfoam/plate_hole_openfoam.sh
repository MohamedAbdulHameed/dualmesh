#!/bin/bash
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# OpenFOAM's classical plateHole case (solidDisplacementFoam): a quarter of a
# square plate with a circular hole, loaded by a uniform traction of 10 kPa.
# The script runs the case, samples sigma_xx along the line x = 0, and exports
# the mesh so that dualmesh can solve the same problem on the same mesh.
#
# Usage:  ./plate_hole_openfoam.sh

if [ -z "$WM_PROJECT_DIR" ]; then
  # shellcheck disable=SC1090
  . "${FOAM_BASHRC:-/usr/lib/openfoam/openfoam2406/etc/bashrc}"
fi
if [ -z "$WM_PROJECT_DIR" ]; then
  echo "OpenFOAM was not found; set FOAM_BASHRC to its etc/bashrc" >&2
  exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
CASE="$HERE/run/plateHole"
TUTORIAL="$FOAM_TUTORIALS/stressAnalysis/solidDisplacementFoam/plateHole"

rm -rf "$CASE"
mkdir -p "$(dirname "$CASE")"
cp -r "$TUTORIAL" "$CASE" || exit 1
cd "$CASE" || exit 1

cp -r 0.orig 0
blockMesh > log.blockMesh 2>&1 || { echo "blockMesh failed"; exit 1; }
solidDisplacementFoam > log.solidDisplacementFoam 2>&1 \
  || { echo "solidDisplacementFoam failed, see $CASE/log.solidDisplacementFoam"; exit 1; }
postProcess -func "components(sigma)" -latestTime > log.sigma 2>&1
postProcess -func singleGraph -latestTime > log.graph 2>&1
foamToVTK -latestTime -ascii > log.foamToVTK 2>&1 \
  || { echo "foamToVTK failed"; exit 1; }
echo "OpenFOAM results are in $CASE"
