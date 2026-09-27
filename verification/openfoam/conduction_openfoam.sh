#!/bin/bash
# SPDX-License-Identifier: LGPL-2.1-or-later
#
# Steady heat conduction in a rectangular plate (the geometry and boundary
# conditions of Example 5.4.2 of Reddy's book) with OpenFOAM's laplacianFoam,
# marched to steady state, and sampled along two horizontal lines.
#
# Usage:  ./conduction_openfoam.sh [nx] [ny]

if [ -z "$WM_PROJECT_DIR" ]; then
  # shellcheck disable=SC1090
  . "${FOAM_BASHRC:-/usr/lib/openfoam/openfoam2406/etc/bashrc}"
fi
if [ -z "$WM_PROJECT_DIR" ]; then
  echo "OpenFOAM was not found; set FOAM_BASHRC to its etc/bashrc" >&2
  exit 1
fi

NX=${1:-40}
NY=${2:-20}
HERE=$(cd "$(dirname "$0")" && pwd)
CASE="$HERE/run/conduction"

rm -rf "$CASE"
mkdir -p "$CASE/0" "$CASE/constant" "$CASE/system"
cd "$CASE" || exit 1

cat > system/blockMeshDict <<DICT
FoamFile { version 2.0; format ascii; class dictionary; object blockMeshDict; }
scale 1;
vertices
(
    (0    0    0) (0.2  0    0) (0.2  0.1  0) (0    0.1  0)
    (0    0    0.01) (0.2 0 0.01) (0.2 0.1 0.01) (0 0.1 0.01)
);
blocks ( hex (0 1 2 3 4 5 6 7) ($NX $NY 1) simpleGrading (1 1 1) );
edges ();
boundary
(
    left   { type patch; faces ((0 4 7 3)); }
    right  { type patch; faces ((1 2 6 5)); }
    bottom { type patch; faces ((0 1 5 4)); }
    top    { type patch; faces ((3 7 6 2)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
DICT

# The temperature on the top edge varies as 500 (1 - 10 x^2).  OpenFOAM's
# codedFixedValue compiles a shared library at run time, which it refuses to do
# when running as root, so the face values are written out as a list instead.
TOP_VALUES=$(python3 -c "
nx = $NX
width = 0.2
print(' '.join('%.10g' % (500.0 * (1.0 - 10.0 * ((i + 0.5) * width / nx) ** 2)) for i in range(nx)))
")

{
  echo "FoamFile { version 2.0; format ascii; class volScalarField; object T; }"
  echo "dimensions [0 0 0 1 0 0 0];"
  echo "internalField uniform 400;"
  echo "boundaryField"
  echo "{"
  echo "    left   { type fixedValue; value uniform 500; }"
  echo "    right  { type fixedValue; value uniform 300; }"
  echo "    bottom { type zeroGradient; }"
  echo "    top    { type fixedValue; value nonuniform List<scalar> $NX ( $TOP_VALUES ); }"
  echo "    frontAndBack { type empty; }"
  echo "}"
} > 0/T

cat > constant/transportProperties <<'DICT'
FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }
DT              [0 2 -1 0 0 0 0] 0.2;
DICT

cat > system/controlDict <<'DICT'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     laplacianFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         5;
deltaT          0.002;
writeControl    runTime;
writeInterval   5;
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
runTimeModifiable true;
DICT

cat > system/fvSchemes <<'DICT'
FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes      { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
DICT

cat > system/fvSolution <<'DICT'
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    T { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; }
}
SIMPLE { nNonOrthogonalCorrectors 2; }
DICT

cat > system/sampleDict <<'DICT'
FoamFile { version 2.0; format ascii; class dictionary; object sampleDict; }
type sets;
libs ("libsampling.so");
interpolationScheme cellPoint;
setFormat raw;
fields (T);
sets
(
    bottom_line
    {
        type    uniform; axis x;
        start   (0 0.0    0.005); end (0.2 0.0    0.005); nPoints 81;
    }
    middle_line
    {
        type    uniform; axis x;
        start   (0 0.05   0.005); end (0.2 0.05   0.005); nPoints 81;
    }
);
DICT

blockMesh > log.blockMesh 2>&1 || { echo "blockMesh failed, see $CASE/log.blockMesh"; exit 1; }
laplacianFoam > log.laplacianFoam 2>&1 || { echo "laplacianFoam failed, see $CASE/log.laplacianFoam"; exit 1; }
postProcess -func sampleDict -latestTime > log.sample 2>&1 \
  || { echo "sampling failed, see $CASE/log.sample"; exit 1; }
echo "OpenFOAM results are in $CASE/postProcessing/sampleDict"
