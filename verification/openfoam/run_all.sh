#!/bin/bash
# SPDX-License-Identifier: LGPL-2.1-or-later
# Run every OpenFOAM cross-verification case and print the comparisons.
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1

"$HERE/cavity_openfoam.sh" "${1:-64}"      && python3 compare_cavity.py --resolution "${1:-64}"
"$HERE/conduction_openfoam.sh" 40 20       && python3 compare_conduction.py --nx 40 --ny 20
"$HERE/plate_hole_openfoam.sh"             && python3 compare_plate_hole.py
