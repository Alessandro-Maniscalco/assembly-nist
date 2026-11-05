# Motor/base fit audit

Goal: distinguish true vendor-CAD interference from grid interpolation before
changing geometry, contact settings or insertion control.

## Source geometry

The source is Hiwonder's `follower arm black.3dm`, obtained from the technical
resources linked by the [official SO101 manual](https://docs.hiwonder.com/projects/LeRobot/en/latest/docs/SO_ARM101_Open_Source_6_Axis_Robotic_Arm_User_Manual.html).
The [vendor folder](https://drive.google.com/drive/folders/1sAwusnQrOkSJRaRBbTIlIXnjiwk_TyVz)
contains Drive file `13ffIl5rWNmyrwV8Q4zCmyH30RVABJGAM`.

CAD SHA256: `f675f6049f94206c7b74ab73c40485913916babff1a6f4fc145dc9040672a2f8`.
The importer preserves cached render triangles from object 260 (base) and
objects 0–13 (motor), converts millimetres to metres and recentres each mesh.
The original assembly transform follows the recorded origins, without the
scripted corridor's additional 0.1 mm lateral offset or 0.24° roll.
OBJ exports match the NPZ triangles used by the grid generator within 5 nm of
text-export rounding. These render meshes are not metrology of physical parts.

## Measured mismatch

Four exact vertical triangle-ray sections at base-frame X = ±10 mm and
Y = 5/10 mm give a **31.800 mm base cavity** and approximately **32.000 mm motor
outer thickness**. In the vendor pose, almost all the approximately 0.200 mm
excess lies at the lower rail; the upper surfaces nearly touch.

| Static pose | Native deepest contact | Largest sampled triangle overlap |
|---|---:|---:|
| Unmodified vendor assembly transform | 0.0959 mm | 0.2047 mm |
| Scripted corridor, seated, +0.24° roll | 0.0461 mm | 0.2139 mm |
| Stalled transport, state 22799 | 0.0810 mm | 0.2129 mm |
| Prepared released final, state 27579 | 0.0663 mm | 0.1614 mm |

Triangle comparisons include all vertices, all face centroids and 4,000
area-weighted surface samples per part. Unsigned distances use the original
triangles; signs use the generator's winding-number convention. Exact winding
checks at contacts and critical samples found no sign disagreements.
These are sampled static diagnostics, not a continuous trajectory certificate.

At the stalled pose's strongest mating contact (14.0 N), the contact point is
0.1455 mm inside the original base and 0.0406 mm inside the motor. The base grid
understates the first distance by approximately 21 µm. Interpolation does add
material at other sampled corners, so its effect is not zero; it does not
explain away the demonstrated source-geometry interference.

## Meaning of native contact distance

For these SDF contacts, native `contact.dist` equals the maximum of the two
SDF values at `contact.pos - contact.normal * contact.dist / 2`. It is neither
their sum at the stored contact point nor maximum mesh penetration. All 108
reconstructed contacts agree within 4.3e-17 m. Therefore a native 0.4 mm safety
gate is a simulator contact metric, not a 0.4 mm whole-mesh certificate.

A vertical pose shift redistributes the local thickness excess. This is not
an exhaustive six-degree-of-freedom impossibility proof. Actual PLA dimensions,
printing tolerances, deformation and insertion forces remain unmeasured; the
audit does **not** establish that the hardware assembly is impossible. It also
does not validate a physical press fit or calibrate MuJoCo's contact compliance.

No CAD cuts, grid refinement, contact-softness calibration or tolerance changes
were performed. Halving both grid spacings would require approximately 480 MB
of new grid files while preserving the actual triangle interference; no such
grids were generated.

## Printable-source cross-check

The vendor `White LeRobot.3mf` archive (Drive file
`1k9OhYWwfIUYxOiEZPKk0ycR2uBvWMZC9`) has SHA256
`2c74eb7ab377c2ffbd1d37c6e381d3483fde1359a5851c7d630e5fdc4dc09710`.
Its base is root object 16, mesh object 15, named `Document_2`, in
`3D/Objects/object_41.model`. Its centred extents uniquely match the render
base within 0.134 µm; bidirectional surface checks confirm identity-frame
alignment without fitted transforms. The meshes have different tessellation,
with sampled surface differences up to 0.173 mm; global equivalence is not claimed.
All four rail sections again measure **31.800 mm**, so the nominal cavity
mismatch also exists in the printable source. This does not prove that the
interference was intentional or that a physical PLA press fit is appropriate.
Component/build transforms and relevant slicer settings are preserved in
`artifacts_local/mating_geometry_audit/printable_source/printable_base_report.json`;
the adjacent `check_printable_base.py` reproduces this read-only check.

## Reproduce and inspect

From the repository root, with the local vendor assets and archived trajectories:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/check_so101_mating_geometry.py
```

Evidence is local and ignored by Git:
`artifacts_local/mating_geometry_audit/audit.json`, `summary.json` and
`vendor_mating_cross_section.png`. The audit records source/model hashes,
individual contacts, both surface comparisons and the four ray sections.
The qualified run used two CPU cores, sampled peak RSS 1.17 GiB and at least
8.28 GiB available host memory. Earlier reports labelled `unqualified` preserve
the corrected contact-convention checks and are not the final evidence.
