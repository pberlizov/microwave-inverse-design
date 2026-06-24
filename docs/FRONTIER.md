# Open-source frontier: inverse design for selective microwave mineral heating

Survey done 2026-06-21 to scope where this thin slice sits relative to existing open
work. The question: **is anyone openly inverse-designing applicator/cavity geometry to
maximise selective absorption (heating) in a target mineral phase vs. gangue?**

Short answer: the *ingredients* are all open and mature, but the *intersection* is not
an established open-source thing. Each piece below exists; the combination is the gap.

## What exists (the ingredients)

### 1. Open EM solvers that run at microwave frequencies
- **openEMS** — free, open-source EC-FDTD solver, the de-facto open microwave FDTD;
  Python/Octave scripting, CSXCAD geometry. The natural production engine here.
  <https://github.com/thliebig/openEMS> · <https://www.openems.de/>
- **FDTDX** — JAX, GPU, memory-efficient autodiff FDTD built *for* inverse design,
  but aimed at 3D nanophotonics. Maxwell is scale-invariant, so it ports to GHz.
  <https://arxiv.org/abs/2412.12360>
- **MEEP** — scale-invariant FDTD with an adjoint/`MaterialGrid` toolchain (what the
  sibling nanophotonics project uses). Works at GHz unchanged.

### 2. Inverse design / topology optimization for microwave EM
Mature for antennas, metasurfaces, and dielectric design — adjoint sensitivity is
standard:
- Inverse design of dielectric materials by topology optimization (DTU).
  <https://orbit.dtu.dk/en/publications/inverse-design-of-dielectric-materials-by-topology-optimization>
- Topology optimization for microwave control with reconfigurable metasurfaces.
  <https://arxiv.org/abs/2311.03018>
- **Closest conceptual neighbour:** inverse design for *near-field radiative heat
  transfer* — an inverse-design problem with a heating-related FOM, but radiative
  transfer between surfaces, not applicator coupling into an ore charge.
  <https://arxiv.org/pdf/1802.05744>

### 3. Microwave-heating simulation (forward, often EM-thermal coupled)
- **QuickWave QW-BHM** — commercial FDTD basic-heating module. <https://www.qwed.com.pl/qw_bhm.html>
- **COMSOL** — commercial multiphysics, the usual EM-thermal workhorse.
- Open EM-thermal coupling is hand-rolled (openEMS field → external thermal solver).
  Forward microwave-heating FDTD studies are common; design *optimization* of the
  cavity for a heating objective is rare and mostly uniformity-driven, not selectivity.

### 4. Microwave processing of minerals (the application physics)
- "Twenty years of … microwave-assisted breakage of rocks and minerals — a review"
  documents the dielectric-contrast heating mechanism (target phases absorb, gangue
  stays cool, differential thermal stress fractures the rock). Experimental + forward
  numerical; **no inverse design.** <https://arxiv.org/pdf/2011.14624>

## The gap (where this project sits)

Two independent searches for inverse-design-of-cavity-geometry-for-selective-mineral-
heating returned **no established open combination**. The people who model microwave
mineral processing (mining/minerals engineering) don't use inverse design; the people
who do EM inverse design work on photonics/antennas. That intersection — *optimise the
applicator so absorbed power concentrates in the target mineral phase* — is the niche.

Caveat (no-hallucination): "no search hits" is not proof of absence. There may be
closed/commercial or non-indexed work. The honest claim is **"not an established,
discoverable open-source capability,"** not "literally first."

## What this thin slice adds vs. the frontier

A minimal, runnable demonstration that **geometry alone moves selective-absorption
contrast** — the assumption everything downstream rests on — using a self-contained 2D
FDFD lossy solver (no MEEP/conda) and the search→evaluate loop pattern reused from the
nanophotonics project. It is intentionally *not* a microwave-heating product: no 3D, no
thermal coupling, no fluidized bed, no measured ore dielectrics. See README limitations.

## If this graduates, the open production path
openEMS (or MEEP/FDTDX for adjoint) for the forward EM, measured ore complex
permittivities from the geophysics literature, an EM→thermal coupling step, and a
geometry/excitation manifold over manufacturable applicator shapes. The relaxed
field-uniformity argument (conductive-particle fluidized beds spreading energy) is what
makes the *selectivity* target — rather than volume uniformity — the right objective and
keeps the optimization low-dimensional and tractable.
