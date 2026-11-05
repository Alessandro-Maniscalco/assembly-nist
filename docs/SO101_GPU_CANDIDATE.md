# Isolated GPU collision candidate

**Goal:** Measure a faster helper-jaw collision candidate without changing the frozen native-physics overnight pilot or original PPO/SimBa/SGS.

**Status, 2026-11-05: not qualified for training.** The pinned Menagerie jaw collisions remove the previous first-step stall, and the nominal contact-operated driver sequence passes on native MuJoCo and MJWarp. Small-part collision fidelity, full-task manipulation/success, and a training adapter remain unqualified. The running pilot and its banks were not changed.

## Candidate and explicit deviations

`scripts/check_so101_gpu_candidate.py` substitutes `replace_gripper_collision_menagerie` only while building scratch XML. Target CAD, nominal threads, physical tool, arm kinematics/inertias/sites, actuators, contact settings and exclusions retain their source definitions. Source: Menagerie commit `3abca2e302625cec9429c49806121ea2855cef81`; local assets/provenance are documented in `SO101_TOOL.md`.

The full candidate has 148 geoms, 65 positions, 58 velocities and ten recognized plugins. Four helper-jaw SDF instances are replaced by the pinned collision pieces/primitives; all remaining task SDFs and the passive button-to-spindle callback are accounted for. The existing GPU importer audits 47 mechanical parameter arrays and relevant solver/state settings. These audits establish what was imported, not engine-equivalent contacts.

The production helper surfaces remain finer 0.2 mm SDF grids. Menagerie is an approximation: independent tool-agent sampling found a moving-finger underfill up to 1.984 mm near a possible small-part contact region. Broad driver pads bridge that region in this fixture. A tool-sequence pass does not qualify horn/screw grasping or prove fidelity to the fine SDF.

GPU line-search iterations were explicitly raised to **100**, retaining native tolerance **1e-8**, main iterations **50**, and all contact parameters. Native remains at its original limit: 20 in the full/thread scene, 50 in the isolated jaw fixture. At 20, the initial full-scene probe reports cumulative overflow bit 1024; at 100, the measured FP64 probes below have zero cumulative overflow flags. No warnings were suppressed. Main workcell timestep is 0.5 ms; the existing thread bench uses 0.1 ms.

Scratch full-scene XML SHA-256: `2f80d553711a293263b3b58b9ab68061e1f7772870565be8dc44ffe764983e67`. Candidate reset signature: `87da1180c6eec00c86d4e90c67a21e0a2f3d08e2e78556197f7219cdae06f8e6`.

## Measurements

Host `dcf-lab`, RTX 5080, MuJoCo/MJWarp 3.17, Warp 1.19.0. The 64-environment native pilot with 16 collection threads and intermittent CUDA PPO, and some CPU layout work, continued concurrently. These short GPU numbers measure captured physics only, excluding environment observations/rewards/resets and PPO.

| FP64 GPU probe | Worlds | Controls/s | Total device memory at final sample |
| --- | ---: | ---: | ---: |
| HOME, loose objects | 16 | 41.81 | 1,275 MiB |
| HOME, loose objects | 64 | 73.83 | 1,627 MiB |
| HOME, loose objects | 256 | 89.19 | 3,131 MiB |
| Cyclic five-task settled fixture | 64 | 37.35 | 1,627 MiB |

A control is 80 physics steps / 40 ms. The HOME runs have short differing warmup durations, so their scaling is indicative. The first candidate synchronized step completed in 11.57 s including loading/compilation, compared with the earlier fine-jaw-SDF probe that did not complete within approximately 235 s. Later cached candidate first steps took around 2 s.

The same candidate's native collection path, five settled task states, 64 environments / four threads, measured **11.46 transitions/s**, 512 valid transitions and zero safety/unstable/done events. This includes observations/rewards and follows the source relative-position zero-action controller; GPU timings hold commands fixed. The native and GPU figures therefore are not a controlled end-to-end training speedup. The much faster single-world initial HOME measurement was unrepresentative of predecessor/thread contact costs.

In the 0.4005 s paired HOME trace, native/GPU positional differences reached 0.164 mm for the motor, at most 0.015 mm for the other nonscrew target parts, and 1.394 mm for the loose rolling screw. Arm position error stayed below 1.25e-6 rad. Free-body angular divergence is retained in the raw traces. No comparison establishes bitwise GPU reproducibility.

## Actual jaw and free tool

The source jaw fixture starts aligned, with the documented +0.6 mm initial tool-X correction. Position commands are OFF 152 encoder ticks, ON 130, OFF 152, OPEN 220; gravity begins after initial closure. No object pose is overwritten after reset, no grasp weld or extra tool actuator is added. This is an already-aligned fixture, not pickup or arm reach.

The final probe records button travel, maximum contact depth and nonfinite state flags **every 0.5 ms physics step**, including inside captured graphs. Both native and GPU pass closure-stays-off, carry-off, press-on, release-off, <3 mm retention drift, <0.3 mm peak contact depth, and >50 mm opening drop. Native/GPU retention drift: **1.662 / 1.652 mm**; opening drop: **207.55 / 207.81 mm**. Native peak penetration is **0.040 mm**; GPU reaches **0.141 mm**. All sampled and monitored states are finite, native warnings and cumulative GPU overflow flags are zero. GPU float32 clock drift reaches 0.188 ms after 3.85 s, without a reset.

The ON button travel and contact counts differ between engines despite matching activation outcome. Spindle angle also diverges; do not describe this as identical mechanics. Raw sequence and trace: `artifacts_local/gpu_menagerie_candidate/tool_monitored_ls100/`.

## Threads and numerical precision

The two-second FP64 guided-rotation probe at line-search 100 is finite, without native warnings or GPU overflow. Native/GPU fitted axial pitch after 0.3 s is **0.49734 / 0.49706 mm per turn**, versus nominal 0.500 mm. The phase residual spans 26.4 / 30.9 µm. A strict 5 µm endpoint criterion is sensitive to the sampling phase: using the first sample after 0.3 s fails on both native (10.28 µm) and GPU (7.67 µm); the previous checker's 0.3121 s endpoint gives 1.68 / 4.92 µm. Both calculations are retained rather than selecting one as conclusive proof. The two-second axial-only 1 N push advances **7.402 / 7.390 µm**, passing the 20 µm axial gate on both engines with zero overflow/warnings and finite trajectories.

`scripts/so101_gpu_fp32.py` is an optional isolated numerical experiment. A first direct FP32 conversion failed: `n - 1.000001` rounded to the final grid index, causing an out-of-bounds boundary sample/NaN, and boundary-gradient branches changed. That failed report is preserved in `sdf_fp32/`.

The current variant keeps grid header decoding, coordinate/cell selection and exterior classification in FP64, with FP32 interpolation and analytic SDF arithmetic. Its 16,000 native-reference queries, including grid boundaries, are finite; maximum distance error is **4.65 nm**, maximum grid-gradient error **1.50e-5**, and maximum analytic-gradient error **0.01895** (female finite-difference gradient). This passes the existing pointwise criteria. Its full-scene collision JIT remained CPU-bound for approximately 80 s and grew to about 5.2 GB process RSS, so that probe was stopped before completion. No mixed-precision physics speed, thread result or training qualification is claimed.

## Reproduce and remaining gates

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u \
  -m scripts.check_so101_gpu_candidate --case tool_sequence --worlds 1 \
  --gpu-ls-iterations 100 --benchmark-controls 0 --out artifacts_local/gpu_candidate_review/tool
```

Use `--case zero --worlds 16|64|256` for a full scene; `--case native_collection --worlds 64 --native-threads 4 --steps 8` creates a scratch five-task fixture without touching training banks. Pass that fixture with `--fixture PATH` for a cyclic GPU load. In heterogeneous runs native trajectory comparison covers world 0; finiteness/overflow checks cover all worlds. `--case thread|axial --steps 20000 --worlds 1 --benchmark-controls 0` uses the native thread bench. `--case sdf --sdf-precision fp32` runs the mixed-precision point probe.

The runner contains a 60 s first-synchronized-step watchdog and a background resource guard that terminates only its own scratch process above 8 GiB total device memory or 12 GiB estimated host use. Earlier raw runs used external timeouts; the mixed compile was manually stopped. GPU benchmark peak recorded memory was 3.13 GB, and the mixed compile approached 11.2 GiB host use. Raw JSON, XML, traces and logs are under `artifacts_local/gpu_menagerie_candidate/`.

Before staging a distinct training run: resolve the small-part jaw fidelity gap; verify target part grasp, contact, placement and nominal task success; qualify full-horizon continuation/reset behavior and environment observations/rewards; then measure complete collection plus learner throughput. The source PPO, SimBa and SGS algorithms remain unchanged.

## Follow-up: measured bottleneck and fine-jaw alternative

`scripts/so101_gpu_profile.py` profiles cached kernels without changing production. On the 64-world, five-task fixture, SDF narrowphase accounts for **92.88%** of measured CUDA kernel time; solver line search is **0.97%** and Cholesky **0.78%**. Reducing solver effort would not address the dominant cost. Timings are CUDA events over eight uncaptured steps; the separate captured benchmarks below exclude profiling and Python launch overhead.

The historical 235 s first-step interruption was **not reproduced**: current fine-jaw HOME probes at line-search 100 complete their first synchronized step in 2.08 s for one world and 2.37 s for sixteen worlds, including cached module loading. Subsequent SDF kernels take about 28 ms/step. The old log already showed cached modules, so compilation was not established as its cause.

Installed MJWarp's mesh–SDF distance query scans every mesh triangle repeatedly during gradient descent. HOME admits left geom 31 / 37 and right geom 68 / 74: `wrist_roll_pitch_so101_v2` (53,994 faces) against the moving-jaw SDF. One isolated left pair costs 25.39 ms at one world. Native [MuJoCo 3.17 mesh–SDF collision](https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_collision_sdf.c#L970) traverses the original mesh faces through a BVH; replacing those faces with a convex hull would change this contact geometry.

A conservative geometric proof supports excluding each wrist body against its own moving-jaw body. The proof encloses every grid cell capable of negative trilinear distance, verifies positive boundary samples, and maximizes the jaw's axial coordinate analytically across the entire gripper angle interval. Wrist roll preserves that coordinate. Both sides have at least **4.487 mm separation** over legal limits, and **1.636 mm** with an additional ±0.05 rad of gripper motion. This covers both wrist collision meshes, including the servo. The proof is specific to the recorded model/grid and legal motion; it does not authorize arbitrary self-collision exclusions.

Native HOME has no contacts between these bodies. Before exclusion, the GPU produces false contacts reaching **2.94 mm penetration** despite the separating bound. Scratch XML adds only the two proved body exclusions; all fine jaw grids, target meshes, tool, controls and other collision pairs remain. The false contacts disappear. Native original/excluded continuations are bit-identical in qpos/qvel across 80 steps for all **53 starts** tested: five independent task starts plus 48 held motor starts, with zero warnings. Other poses that admit mesh–SDF pairs still need independent contact parity.

| Fine-jaw scratch fixture, FP64 | Worlds | Captured physics transitions/s |
| --- | ---: | ---: |
| Five independent task starts, cyclic | 64 | 37.35 |
| 48 near-held motor starts, cyclic | 64 | 54.57 |

Both four-control benchmarks remain finite with zero overflow flags. Peak recorded probe RSS is 3.55 GB and total device memory 1.85 GB; native training continues concurrently. These are fixed-control physics measurements, not complete environment/PPO throughput or successful assembly. The fine geometry avoids the Menagerie fidelity compromise, but the five-task rate offers no compelling production migration benefit.

After the exclusions, neither fixture has a mesh–SDF broadphase pair. A separate per-pair diagnostic measures table/base at 1.84 ms for 64 worlds, female/male at 1.42 ms for 25 worlds, and several male-thread/part candidate pairs at 1.08–1.41 ms. Some candidates produce zero final contacts but still incur SDF search. These timings isolate pairs and must not be added as an exact decomposition of the simultaneous kernel. Remaining work is target/thread SDF search and arithmetic, not high-face helper meshes or solver tuning.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.so101_gpu_profile \
  --geometry fine --mode separation --out artifacts_local/gpu_profile/separation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.so101_gpu_profile \
  --geometry fine --exclude-separated --worlds 64 --bank-dir data/so101/banks_near_v1 \
  --benchmark-controls 4 --out artifacts_local/gpu_profile/fine_benchmark
```

Use `--bank-selection motor-near` for held starts or `--mode sdf-pairs` for isolated pair timing. Default guard limits are 4.5 GiB process RSS, 2 GiB minimum available host memory, 8 GiB total GPU memory and 90 s wall time. Evidence is under `artifacts_local/gpu_profile/`; no production scene, bank or learner was changed.

A final bounded mixed-precision attempt reused the already validated point-query code unchanged, with the fine jaws and proved exclusions. The explicitly raised limits were 5.5 GiB RSS and 180 s. Compilation crossed the RSS threshold between one-second watchdog samples: the terminating sample was **6.02 GiB RSS**, **2.10 GiB host memory available**, and **1.80 GB total device memory**. The guard terminated only that probe (exit 70), before a completed physics step. `fine_mixed64/profile.json` preserves the failure. No mixed-precision throughput or mechanical parity is available; further compilation and a GPU environment port were stopped. The native learner continued.

The standalone mixed query module defaults to backward code generation, but that does not explain this failure: the compiled collision module already disables it. Its retained 503,880-byte CUDA source contains one forward narrowphase kernel, no backward kernel and no query kernel; the cached FP64 counterpart is 504,038 bytes. Warp propagates backward use from the calling kernels. `backward_codegen_audit.json` and `mixed_failed_codegen.cu` preserve that check; only this probe's interrupted temporary cache directory was removed. No repeat compilation was attempted.

### Mixed precision after freeing host memory

A separate quiet compilation window completed the unchanged mixed SDF collision
kernel in 86.18 seconds after the prior learner and CPU probes had checkpointed
and stopped. The guard allowed at most 9 GiB process RSS, required 2 GiB available
host memory, capped total GPU memory at 8 GiB, and limited the attempt to 240 s.
Post-compilation RSS was about 3.74 GiB; the measured run used at most 1.35 GiB GPU
memory. The earlier under-load memory failure remains preserved.

With fine jaw SDFs and the proven wrist/jaw exclusions, the 64-world five-task
fixture measured **102.07 raw physics transitions/s** over one captured control
block, finite with zero overflow. SDF collision still occupied 84.6% of kernel
time. This short result motivates further qualification; it does not establish
end-to-end learner speed or mechanical equivalence. The native v2 learner was
then started while actual contact/control/reset checks proceed separately.
Evidence: `artifacts_local/gpu_profile/quiet_mixed/profile.json`.
