# Gripper-operated screwdriver prototype

Goal: carry a screwdriver, press its physical button with the existing SO101
gripper, and release the button while retaining the tool. This is an unmeasured
accessory design and contact simulation, not production CAD or hardware proof.

## Basis and design

[HOTO's S-A01 page](https://hototools.com/products/snapbloq-a01) specifies
0.05/0.35/0.5 N m electric torque and push-button operation. We model the lowest
setting. A later-found [official HOTO Japan specification](https://hototools.jp/products/hfe0025gl)
lists QWLSA002 at approximately 181.7 × 19.7 × 22.5 mm, 105 g, 250 rpm, with
C4 × 28 mm bits. Its [manual](https://hoto-jp.com/document/HFE0025GL_HOTO_Electric_Precision_Screwdriver_Master_Set.pdf)
confirms the controls. The frozen pilot's dimensions below remain a proxy,
not a fitted HOTO sleeve. Bit inclusion in the stated length, detailed button
positions/force/travel, inertia and CAD tolerances still need verification.

[Schmidbauer et al., FAIM 2022](https://link.springer.com/chapter/10.1007/978-3-031-18326-3_17)
demonstrated a printed holder that lets a two-jaw gripper retain and operate a
commercial screwdriver. Their wedges and rack-driven lock support the general
approach, not this SO101 implementation. Bare-button pinching couples holding
force to activation; copying their complete holder adds size and parts. Our
simpler candidate uses a split sleeve and spring-loaded saddle.

In `assembly_policy/so101/tool.py`, grey is the nominal tool, orange the fitted
sleeve, and blue the movable saddle. Local +Z points toward the bit; +Y is the
button side. A pusher contacts the separate native forward button; reverse stays
exposed. The passive slide represents two guide pins and captive travel stops.
Proposed manufacture uses two clamped printed shells, friction liners, a saddle,
two guide pins and return springs. Bores, fasteners and manufacturing tolerances
still require dimensioned CAD based on the real tool.

| Quantity | Nominal design |
| --- | --- |
| Case | 20 × 20 × 156 mm, rounded rectangle |
| Sleeve maximum width / axial length | 28 / 50 mm |
| Opposing contact pads / unpressed jaw span | 26 × 26 / 33 mm |
| Saddle stroke / clearance before button contact | 2 / 1 mm |
| Spring preload / stiffness | 2 N / 400 N m⁻¹ total |
| Native button stroke / activation | 1 / 0.6 mm |
| Grasp center / tip in tool frame | (0, 1.5, 15) / (0, 0, 105) mm |
| Complete modeled mass | 132.5 g |

Activation requires approximately
`2 + 400*(0.001+0.0006) + 1400*0.0006 = 3.48 N`.
The clearance provides compliant OFF travel. Continuous gripper position targets
are necessary to select carry and squeeze apertures. There remain twelve robot
actions, no electrical trigger channel and no grasp weld. Only native button
displacement starts the spindle.

## Verified scope and limits

The full, unchanged workcell now completes a scripted table pickup, at least
40 mm lift, physical button activation, deactivation, table return and release.
`assembly_policy.so101.tool_demo` uses a mirrored grasp, 2 mm lateral approach
clearance, a 10 mm higher grasp, short Cartesian waypoints, and 0.4 s spindle
coast-down. The 40,601-step trace has no warnings and 0.1584 mm peak penetration;
the spindle reaches 25.92 rad/s and falls to 0.0052 rad/s after coast-down.
This is one nominal pose. The tool shifts several millimetres within the grasp;
screw engagement, reaction torque and repeatability still require qualification.

```bash
.venv/bin/python -m assembly_policy.so101.tool_demo --out artifacts_local/tool_pickup/check
MUJOCO_GL=egl .venv/bin/python -m assembly_policy.so101.trajectory artifacts_local/tool_pickup/check.npz
```

The ideal-jaw test retains a free tool OFF at 2.8 N, including a 0.05 N m reaction
load, and presses ON at 4 N (0.889 mm button travel, 25.92 rad/s). Releasing to
2.8 N turns it off. Maximum four-second drift is 0.80 mm; opening produces a
294 mm fall. Reproduce with:

```bash
.venv/bin/python scripts/check_so101_tool.py --out artifacts_local/so101_tool/bench_check.json
```

Actual source SO101 jaws also pass an aligned gravity carry/press/release check
with position gain 998.22 and a conservative, assumed 0.35 N m cap. This is not
the manufacturer's rating: [WowRobo's C018 page](https://shop.wowrobo.com/products/feetech-sts3215-servo-12v-30kg-high-torque-servo-for-so-arm100)
lists approximately 30 kg cm (2.94 N m) stall torque at 12 V. Commands use the
[4096-step servo resolution](https://www.waveshare.com/st3215-servo.htm?sku=22414).
The palm is fixed; this does not test pickup.

| Tested target ticks | Radians | One-second gravity result |
| --- | --- | --- |
| 130 / 135 | 0.199418 / 0.207087 | Retained, ON |
| 140 / 145 / 150 | 0.214757 / 0.222427 / 0.230097 | Retained, OFF |
| 155 / 160 | 0.237767 / 0.245437 | Drops |

The nominal **145 → 130 → 145** sequence never activates during initial closure
(maximum 0.190 mm), presses to 0.998 mm/25.92 rad/s, then releases to approximately
0 mm/0.002 rad/s. Maximum drift is 1.634 mm and penetration 0.244 mm. Opening
releases the free tool. These are tested target points, not hardware calibration.

```bash
.venv/bin/python scripts/check_so101_tool_gripper.py --sequence --out artifacts_local/so101_tool/actual_jaw_check.json
```

**Torque robustness fails.** Imposed 0.05 N m case torque while OFF145 brings the
button to 0.598 mm, almost activating it. OFF148 and a mirrored, 1 mm more distal
grip actually reactivate (0.796/0.805 mm). Do not extend the ideal-jaw result to
SO101 qualification. The pilot geometry is frozen; locating-rib experiments stay
in scratch scenes. Pickup, orientation changes, bit engagement and assembly
success require separate validation.

At a scratch 1.0 N m cap, the old direct sequence activates during initial
closure (0.898 mm). Wider OFF targets 150/152 avoid that transient but slip
7.31/10.48 mm after release. A 25 Hz encoder-tick ramp keeps closure OFF and
switches correctly, but still reaches 4.84 mm drift and 0.339 mm penetration.
None passed all existing gates; raising the cap requires separate tool-control
qualification. Production remains at 0.35 N m.

## Jaw collision correction

The original single convex jaw hull causes 9.86 mm false penetration at the
aligned pose. A bounded [CoACD](https://github.com/SarahWeiii/CoACD) attempt produced
no completed mesh in 14.5 minutes. The selected fallback,
`scripts/so101_gripper_sdf.py`, samples untouched watertight STL into 0.2 mm SDF
grids 201/202, using 126 MB total. The aligned pose then has zero penetration.
Across 20,000 surface/volume samples per jaw, maximum errors are 68/93 µm;
critical finger-region errors are 63/60 µm. These are sampled checks, not a
worst-case certificate. The local manifest records source hashes and errors.
`collision.replace_gripper_collision` changes only collision geometry; visual
CAD and body inertias stay intact.

An optional [pinned Menagerie collision model](https://github.com/google-deepmind/mujoco_menagerie/tree/3abca2e302625cec9429c49806121ea2855cef81/robotstudio_so101)
is available through `menagerie_collision.py`, without importing its different
TCP, camera, contact settings or controls. Its nominal 152→130→152 tool check
passes after clearing a 0.55 mm fingertip-proxy protrusion. It remains approximate:
sampled distal overfill reaches 0.68 mm and moving-finger underfill 1.98 mm;
denser local sampling finds 2.30 mm omissions. Two minimal patch attempts failed
coverage or added-overfill gates and remain unpromoted. Run
`python -m scripts.check_so101_menagerie_collision --out artifacts_local/so101_tool/menagerie_comparison.json`
for the comparison and plot. The running pilot continues to use the fine SDFs.
