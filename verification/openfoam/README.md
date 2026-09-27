# Cross-verification against OpenFOAM

These scripts run [OpenFOAM](https://www.openfoam.com) and dualmesh on the same
problems and compare the results. They need an OpenFOAM installation; source
its `etc/bashrc` first, or set `FOAM_BASHRC`.

```console
. /usr/lib/openfoam/openfoam2406/etc/bashrc

./cavity_openfoam.sh 64        && python compare_cavity.py --resolution 64
./conduction_openfoam.sh 40 20 && python compare_conduction.py --nx 40 --ny 20
./plate_hole_openfoam.sh       && python compare_plate_hole.py
```

| case | OpenFOAM solver | what is compared |
| --- | --- | --- |
| lid-driven cavity, Re = 100 | `icoFoam` | centreline velocity profiles, also against Ghia et al. (1982) |
| steady conduction in a plate | `laplacianFoam` | temperature along two lines, on the same grid |
| plate with a hole | `solidDisplacementFoam` | stress along the symmetry line, **on OpenFOAM's own mesh** |

The third case imports the mesh that OpenFOAM generated, so the two codes solve
the same discrete geometry with the same boundary conditions.

Everything the scripts produce goes into `run/`, which is not tracked by git.
A summary of the results is in `docs/openfoam.rst`.
