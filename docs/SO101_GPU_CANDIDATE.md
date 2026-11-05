# Optional GPU physics qualification

**Goal:** Qualify isolated GPU collision variants while preserving native reference evidence and the original PPO/SimBa/SGS.

**Status, 2026-11-05:** The whole-helper SDF backend passes geometry, tool, environment and a five-update 128-world PPO pilot at the explicit float32 tolerance 1e-6. It is selected for the bounded overnight run; native remains the default/reference. Earlier Menagerie, raw-helper-mesh and 1e-8 solver failures are preserved below. No result establishes learned loose-part assembly or hardware transfer.

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

### Fine-geometry environment qualification: faster, but not ready

`assembly_policy/so101/warp_env.py` provides an isolated optional rollout adapter;
the production factory and learner do not import it. It reuses the native
environment's observations, twelve relative joint controls, costs, task success,
timeouts and safety checks. It calls supported `mjw.reset_data` before each
control block, restores FULLPHYSICS, and records every fine-step state and contact
sensor. Raw time must match iterative float32 additions exactly before returning
the environment's float64 clock. Sensor matching capacity is explicitly 512;
each sensor value is independently checked against the minimum GPU contact
distance. GPU line search remains explicitly 100, versus native 20.

The two wrist/jaw exclusions are recomputed from the actual fine geometry, with
a per-step joint-domain guard. Every compiled numeric model array and all solver
options are compared against the caller's native model, allowing only those
exclusions and associated name-table metadata. Runtime mass, timestep,
integrator and sensor mutations are rejected; an unexpected sensor cannot be
substituted for the global minimum-distance contact monitor.

At **16 worlds**, 40 control blocks pass all functional checks: ordinary actions,
observations and outcomes; partial SGS resets and history; two-control timeout;
held-versus-released motor, horn and retaining-frame completion; independently
audited eleven-pose stability windows; physical velocity safety; rejection of
nonfinite inputs and out-of-domain jaws. The six held starts remain nonterminal;
all ten opening/already-released cases succeed without safety termination. Every
fine-step sensor audit and raw-clock check passes, with zero overflow. These are
prepared task starts, not learned loose-part assembly.

| Complete environment collection | Native, 16 threads | Mixed GPU |
| --- | ---: | ---: |
| 16 worlds, four timed controls | 15.92 transitions/s | 44.87 transitions/s |
| 64 worlds, four timed controls | 24.16 transitions/s | 62.21 transitions/s |

Both include endpoint native forwards, observations, rewards and safety under
concurrent native training; neither includes a PPO update. The 64-world benchmark
finishes before its later reset-source preparation exceeds the 5 GiB probe RSS
cap. Preloading those sources before GPU allocation allows the complete 16-world
suite to finish. These short measurements do not establish sustained learner
throughput or memory requirements.

Strict repeatability fails separately from reset hygiene. Deliberately
contaminated forces, acceleration and warm starts are cleared by supported
`reset_data`; the first repeated fine step differs by at most 6.98e-10 in qpos
and 3.20e-6 in qvel. Over 80 steps, loose screw contact jitter amplifies this to
about 0.9 rad/s angular-velocity difference, while part positions differ by at
most 17.2 µm and outcomes remain equal in the diagnostic. Canonical contact and
constraint ordering does not remove the jitter. Full invariant-atomic ordering
fails with a Warp workspace dimension exceeding its signed-32-bit limit. No
global deterministic workaround is used. A discarded blanket-array-zeroing
experiment also erased structural Data fields and is explicitly invalid evidence.

The nominal multi-turn thread fit is 0.497451/0.496923 mm per turn for native/GPU,
against 0.500 mm nominal; both pass the existing 5 µm slope criterion. A 1 N
axial-only test settles 6.94/6.39 µm. The shorter native thread-fit failure remains
preserved. Abrupt fine-jaw tool actuation and a separate 0.2 s, 25 Hz ramp retain
OFF/ON/OFF/drop behavior, but GPU peaks of 0.3081 and 0.3189 mm fail the unchanged
0.300 mm tool-bench limit.

Most decisively, replaying all **40,601 original twelve-arm control samples**
without intermediate resets preserves pickup, OFF carry, physical ON activation,
coast-down and return in four GPU worlds, but produces **6.731 mm penetration**
during final withdrawal. Earlier stages stay below 0.159 mm. Static queries at
the saved GPU poses identify false mesh–SDF contacts: right shoulder versus fixed
jaw (3.210 mm), and right upper arm versus fixed jaw (5.040 mm); native finds no
contacts for either pair at those same poses. Recorded arm velocity reaches
26.68 rad/s. These nonadjacent links can potentially collide elsewhere, so they
have not been excluded. The contact/velocity guards would terminate such a
training transition, but they do not repair the incorrect contact physics.

**Readiness remains NO-GO** until these mesh–SDF artifacts are resolved and
requalified. Evidence and the concise handoff are under
`artifacts_local/warp_env/qualification_summary.json`; the complete functional
report is `environment16/environment.json`, and failed full-motion evidence is
`tool_replay/` and `withdrawal_contacts/`. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u \
  -m scripts.check_so101_warp_env --case environment --worlds 16 \
  --out artifacts_local/warp_env/environment_review
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u \
  -m scripts.check_so101_warp_env --case tool_replay --worlds 4 \
  --out artifacts_local/warp_env/tool_replay_review
```


## Whole-helper SDF variant: tool and environment gates passed

The separate helper-grid candidate replaces all 30 active helper mesh colliders
with eleven 0.15 mm source-surface SDF grids. Fine jaw grids 201–202, Hiwonder
surfaces, nominal threads and physical driver remain unchanged. It adds **no
collision exclusions**. `helper_sdf/handoff.json` records the native geometric
and physical qualification, including the approximation's limits.

Direct import of archived model `214c2832347e3419dcec76afe92754193fb84d32e589c567b90e89588111c8b3`
passes the original 40,601-step, 20.3005-second tool replay in **all four GPU
worlds**, without intermediate resets. Pickup exceeds 40 mm; carry stays OFF;
physical squeezing activates 25.92 rad/s; release/coast stops the spindle; return,
release and empty-hand withdrawal pass. Maximum penetration is **0.165900 mm**
against the unchanged 0.300 mm gate; maximum helper speed is 0.578592 rad/s.
All contacts are monitored at every fine step, with zero overflow and exact
repeated-float32 raw clocks. The earlier 6.731 mm mesh–SDF artifact disappears.

The sensor-equipped archive `c69d5fe2201b004004dd60e14de9f684f64e11bf6edc33b8fbdb094a7724753b`
passes a sensor-only audit of every existing numeric model array, option and
semantic name. The loader pins its MJB, plugin binaries, helper manifest and all
18 referenced grid hashes. `archived_helper_factory` plus explicit
`model_variant='whole_helper_sdf'` imports this model without rebuilding canonical
XML. Line search remains 100; minimum-contact sensor capacity is 512.

At **64 worlds**, ordinary actions/outcomes, partial reset/history, timeout,
nonfinite guards, physical velocity safety, all fine-step sensor/clock checks,
and prepared motor/horn/retainer completion pass. All 24 held controls stay
nonterminal; all 40 opening/already-released cases succeed without safety failure.
Sixteen timed controls including observations/rewards measure **66.87 GPU versus
27.90 native transitions/s** (2.40×) under concurrent native training. This is
collection throughput, not end-to-end PPO speed. Total GPU usage was about 4.12 GB.

Supported reset-data contamination checks and ordinary five-frame history shifts
pass. Bitwise repeatability still fails: repeated part positions differ by up to
12.1 µm, while loose screw angular velocity differs by up to 1.36 rad/s. No
deterministic workaround is enabled. The older isolated tool bench's 0.308/0.319
mm depth failures remain preserved; this pass qualifies the actual recorded
pickup protocol, not arbitrary squeeze trajectories.

Evidence: `artifacts_local/whole_helper_gpu/{replay,model,environment64_r2,reset64}`.
The first environment attempt's API typo is retained in `environment64.log`;
the corrected complete run is `environment64_r2`. Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.check_so101_warp_env \
  --case tool_replay --worlds 4 --replay artifacts_local/helper_sdf/native_tool_replay \
  --candidate-mjb artifacts_local/helper_sdf/native_tool_replay.mjb --out artifacts_local/whole_helper_gpu/replay_review
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.check_so101_warp_env \
  --case environment --worlds 64 --controls 16 --bank-dir data/so101/banks_near_v2 \
  --candidate-mjb artifacts_local/whole_helper_gpu/model/candidate_sensor.mjb --out artifacts_local/whole_helper_gpu/environment_review
```

## Learner integration and preserved convergence failure

Install optional probe dependencies with `pip install -e '.[so101,so101-gpu]'`; the complete toolchain is tested on Python 3.12. Trainer flags `--so101-physics whole_helper_sdf --so101-physics-model PATH.mjb` explicitly select the archived model. Preflight and supervisor use corresponding `--physics` and `--physics-model` flags. Model, plugins, grids, adapter source and runtime versions are pinned separately from the canonical reset-bank signature. Evaluation intentionally uses canonical native physics and records both backends.

Two 64-world, 32-step PPO pilots use the final native v2 weights with fresh optimizer/SGS. Both complete one finite update at about 62 transitions/s, then abort before their next update on `OverflowType.ITERATIONS` (512). Increasing the GPU solver cap from 50 to 200 at unchanged tolerance 1e-8 does not resolve the contact. Line search stays 100, and overflow checks remain enabled. The second failure saves all initial states, controls, fine-step states, sensors and flags in `runs/so101_gpu_v2_20261105/physics/failure_000036.npz`; original logs remain in both pilot directories. This prevents unattended continuation with that 1e-8 GPU runtime despite the narrower functional passes above.

## Explicit float32 solver tolerance and repeated qualification

Only world 54 exhausts the second pilot's solver. An instrumented replay reproduces the failure at its first fine step: iterations 3–200 cycle, with scaled improvement about 6.2–7.4e-7 and scaled Newton model improvement about 2.3–2.5e-7. Increasing the cap further cannot establish convergence in this trace. States remain finite, penetration stays below 0.058 mm, and helper speed stays below 3.01 rad/s. Native physics at 1e-8 needs at most ten iterations across all six constraint islands over the 80-step replay, with no cap hits or warnings. Earlier diagnostic fields reporting three native iterations cover island 0 only; the separate all-island audit preserves that correction.

MJWarp 3.17's installed `io.py:469–472` defaults GPU tolerance to at least **1e-6** for float32 arithmetic. The optional adapter now retains that upstream numerical setting, with solver cap 200, line-search cap 100 and every overflow check enabled. Native physics remains at **1e-8**. This is an explicit numerical deviation, recorded in runtime identity and reports; it does not establish GPU convergence at 1e-8. Replaying the failed world at 1e-6 needs at most seven iterations, produces zero overflow, and differs from native part positions by at most 5.62 µm over 40 ms. It is a single-world comparison, with ordinary GPU nondeterminism still present.

At this setting, the actual 40,601-step tool replay passes all eight physical gates in all four worlds, with maximum penetration **0.152565 mm**. Complete finite state/control archives and the final PASS report were saved; the surrounding shell subsequently reported exit 143 without a watchdog record. That unexplained post-result termination is retained in the handoff. The 64-world environment process exits normally: all 40 opened/released cases complete, all 24 held controls remain nonterminal, and action/history/reset/timeout/contact/velocity/nonfinite checks pass. Strict repeatability remains unqualified. Sixteen five-task controls measure 67.11 GPU versus 26.36 native collection transitions/s under concurrent native training.

A separate matched scaling probe repeats the same 64-state mixture of 32 independent loose starts and 32 prepared TRAIN holds. Each size runs eight measured controls including observations, rewards and terminal/reset handling, excluding actor/PPO updates. All runs exit normally with zero safety events, overflow and sensor-audit error.

| Worlds | Collection transitions/s | Peak process RSS, GiB | Total GPU memory at final sample, MiB |
| --- | ---: | ---: | ---: |
| 64 | 51.67 | 4.18 | 4,115 |
| 128 | 56.02 | 4.61 | 4,531 |
| 256 | 53.59 | 5.47 | 5,299 |

These short measurements favor 128 worlds only modestly; the orchestrator selected 128 for a fresh actual PPO pilot. They do not establish a learned assembly rate or justify extrapolating throughput to larger batches. Evidence and the unchanged runtime hashes are indexed in `artifacts_local/whole_helper_gpu_tol1e6/handoff.json`. Reproduce the solver case with `scripts/check_so101_gpu_solver.py`; `scripts/check_so101_warp_env.py --case collection` provides the matched scaling probe. Use fresh output directories to preserve the earlier failures.

## Passing learner pilot and bounded run

`runs/so101_gpu_tol1e6_20261105` starts from the native v2 final iteration 112 weights with fresh optimizer/SGS. Five complete updates produce 20,480 transitions and 153 episodes, including 45 prepared-TRAIN successes, with finite losses, zero solver overflow and zero numerical failures. Measured end-to-end throughput is 61.27–66.80 transitions/s. The final checkpoint was verified before supervisor resume through 14:45 UTC; independent reviews use canonical native physics. Pilot evidence is `pilot_result.json`; this is not a loose-assembly success benchmark.

To recreate the archive after the canonical assets, bank and scripted tool trace are available:

```bash
.venv/bin/python -m pip install -e '.[so101,so101-gpu]'
.venv/bin/python scripts/so101_helper_sdf.py
.venv/bin/python scripts/check_so101_helper_sdf.py --case geometry
.venv/bin/python scripts/check_so101_helper_sdf.py --case frames
.venv/bin/python scripts/check_so101_helper_sdf.py --case states
.venv/bin/python scripts/check_so101_helper_sdf.py --case replay \
  --replay artifacts_local/tool_pickup/raised10_coast
.venv/bin/python scripts/check_so101_warp_env.py --case prepare_candidate \
  --replay artifacts_local/helper_sdf/native_tool_replay \
  --out artifacts_local/whole_helper_gpu/model
```

Inspect each report, repeat the tool/environment/runtime checks above, and run a fresh learner pilot before unattended use. Native and GPU geometry are approximations with separate provenance; larger batches and exact GPU continuation are not automatically qualified.

## Endpoint forwarding and behavioral review

The first bounded GPU continuation stopped cleanly at iteration 68 after the
14:01 frozen iteration-60 native loose check produced 0/10 successes and ten
safety terminations. It had no numerical/solver failures. This behavioral
regression is separate from engine qualification; see [status](SO101_STATUS.md).

Profiling then isolates native endpoint `mj_forward` as about 0.57 s of a
0.59 s serial refresh for 128 worlds. Eight independent native-data workers
reduce the endpoint to about 0.091 s; unchanged observation packing takes about
0.011 s. All 16 frozen and 28 live integration checks match exactly, including
contacts/forces, sensors, success/stability, history and partial resets. Native
environments remain serial; the GPU adapter retains `endpoint_threads=1` as a
fallback and records the selected count plus `env.py` in its runtime identity.
This is an endpoint-only speedup until a complete learner is timed. Initial
profiling exceeded its strict 3 GiB limit by 55.6 MiB; the separately bounded
production qualification peaks at 3.12 GiB under a 4 GiB limit, with ample host
headroom. Reports remain under `artifacts_local/refresh_profile/`.
