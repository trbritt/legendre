# legendre

[![CI](https://github.com/trbritt/legendre/actions/workflows/ci.yml/badge.svg)](https://github.com/trbritt/legendre/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/legendre.svg)](https://crates.io/crates/legendre)
[![docs.rs](https://docs.rs/legendre/badge.svg)](https://docs.rs/legendre)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**A block-structured, deterministic, scheduler-driven PDE simulation framework in Rust.**

`legendre` solves systems of time-dependent partial differential equations — deterministic or stochastic, in any spatial dimension — on block-decomposed structured grids, with zero-cost abstractions separating the *mathematics* (models, operators) from the *numerics* (discretization policies, integrators) from the *execution* (schedulers, storage, observation).

Phase-field solidification, reaction–diffusion, and heat transport are *models* here, not the framework: the same trait surface that runs a 100-million-cell dendritic nucleation study runs a 3D heat equation from ~60 lines of model code.

```
┌─────────────────────────────────────────────────────────────────┐
│  ∂φ/∂t = ∇·[A²∇φ − AA′∂⊥φ] + φ − φ³ − λu(1−φ²)² + b·ξ(x,t)      │
│  ∂u/∂t = D∇²u + ½ ∂φ/∂t                                         │
│                                                                 │
│  · 101,606,400 cells · 6,400 blocks · bit-reproducible ·        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table of contents

- [Design principles](#design-principles)
- [Architecture](#architecture)
- [The trait hierarchy](#the-trait-hierarchy)
- [The timestep](#the-timestep)
- [Determinism guarantees](#determinism-guarantees)
- [Observation pipeline](#observation-pipeline)
- [What is included](#what-is-included)
- [Quick start](#quick-start)
- [Rendering results](#rendering-results)
- [Validation](#validation)
- [Performance](#performance)
- [Scope: what this framework can and cannot solve](#scope-what-this-framework-can-and-cannot-solve)
- [Roadmap](#roadmap)
- [License](#license)

---

## Design principles

The architecture follows four rules, enforced by the type system rather than by convention:

> **1. Mathematical objects own no execution.**
> A `Model` cannot spawn threads. A `Grid` cannot write files. An `Operator` cannot allocate. If a trait's job is mathematics, its signature cannot express side effects.

> **2. Execution is scheduler-driven.**
> Everything runs because the `Scheduler` dispatches it. Exactly one module in the crate is permitted to name Rayon; a `SerialScheduler` is the semantics oracle that every parallel scheduler must reproduce **bit-for-bit** (and tests enforce this).

> **3. Storage is separate from views.**
> Fields are typed views into opaque storage backends produced by an `Allocator` exactly once, at setup. Nothing allocates in the hot loop — scratch memory is a worker-pinned pool, integrator stages are pre-allocated, and snapshot buffers cycle through a fixed ring.

> **4. Numerical methods are policies.**
> `Grid + Discretization → Operators`. A model states *what* it needs (`D: Discretizes<G, Laplacian>`) and never learns *how* it was realized. Swapping second-order finite differences for a finite-volume scheme changes one type parameter, zero model code.

---

## Architecture

### Ownership graph

Every object has exactly one owner and one responsibility:

```
                        Simulation
                            │
        ┌────────┬──────────┼──────────┬───────────┬─────────┐
        │        │          │          │           │         │
    Scheduler  State      Grid   Discretization  Model   Integrator
        │        │          │          │           │         │
        │    ┌───┴───┐   Blocks    Stencils     V₀(Y,t)   stages,
     blocks  │Storage│      │          │        Vⱼ(Y) noise axpy
      → CPU  │ Views │   topology  operators                 │
             └───────┘                                       │
        ┌────────────┐                                       │
        │ Observers  │ ◄── notified after each completed step┘
        │ (async)    │
        └────────────┘
```

### The fundamental unit is the block, not the grid

Even a uniform Cartesian grid is a collection of congruent blocks. This one decision buys cache locality, natural parallel work units, perimeter-local halo exchange, and an execution model that is *unchanged* when adaptive mesh refinement arrives — refinement replaces one block with children; the scheduler never notices.

```
   Grid                    one Block (ghost-inclusive slab)
   ┌────┬────┬────┐        ╔═══════════════════╗
   │ B0 │ B1 │ B2 │        ║ g  g  g  g  g  g  ║   g = ghost ring
   ├────┼────┼────┤        ║ g ┌─────────────┐ ║       (halo-exchanged from
   │ B3 │ B4 │ B5 │  ───►  ║ g │  interior   │ ║        neighbors, or filled
   ├────┼────┼────┤        ║ g │  cells      │ ║        by the model's BCs)
   │ B6 │ B7 │ B8 │        ║ g └─────────────┘ ║
   └────┴────┴────┘        ╚═══════════════════╝
```

State is stored **block-major** (`blocks × fields`): each block bundles every field slab it owns, so the scheduler hands each worker a structurally disjoint `&mut BlockStorage` — parallel mutation with **no interior mutability and no `unsafe`**.

### Storage → views

```
   Allocator ──alloc once──►  StorageBackend  ──borrow──►  G::View<'_, T>
   (system heap today;        (opaque slab:                (typed, ghost-aware,
    pools / mmap / GPU         DenseStorage,                grid-associated GAT;
    later, zero trait          later Blocked/               monomorphizes to a
    changes downstream)        Adaptive/Gpu)                single indexed load)
```

Views are **grid-associated types** (GATs): the grid wraps a raw slab in a view that knows the block's extent and ghost width. Under AMR, a grid can hand out views that transparently handle coarse–fine interpolation — without any stencil signature changing.

---

## The trait hierarchy

| Trait | Owns | Cannot |
|---|---|---|
| `Grid` | topology, block layout, index→coordinate maps, typed views | hold field data, do IO |
| `StorageBackend` / `Allocator` | bytes; the single allocation point | know arithmetic exists |
| `State` | named fields over blocked storage; vector-space ops (`axpy`, `copy`, noise) | know physics or schemes |
| `Stencil<G>` | one operator realization on one block (ghost width, apply) | allocate, know physics |
| `Discretizes<G, Op>` | *policy*: realize mathematical operator `Op` on grid `G` as a stencil | — (open universe: new operators break nothing) |
| `Model<G, D>` | the PDE: fields, RHS, noise amplitude, boundary conditions | mutate state, see `dt`, spawn work |
| `Integrator<G, D>` | the timestep: stage buffers, `dt` scaling (incl. `√dt` noise) | index space — stages are pure vector algebra |
| `Scheduler` | dispatching disjoint block work with worker-pinned scratch | change results (bitwise identity enforced) |
| `Observer` / `SnapshotSink` | consuming completed steps / snapshots | block the solver |

The policy pattern in one picture:

```
        Grid              +        Discretization        →      Stencil
   CartesianGrid<2>              FiniteDifference              CentralLaplacian
   CartesianGrid<2>              FiniteVolume                  KarmaRappelFlux
   CartesianGrid<3>              FiniteDifference              CentralLaplacian
   QuadTree (future)             FiniteVolume                  (coarse–fine aware)

   Model:  impl<G, D> Model<G, D> for Diffusion
           where D: Discretizes<G, Laplacian>          ← states *what*, never *how*
```

**Dimension is an associated fact, not a trait parameter.** Concrete grids carry `const D: usize`; generic solver code never touches it. The 2D and 3D heat models in this repository are the same source text.

---

## The timestep

`Model::step()` does not exist. Models expose `rhs` (dY/dt) and never mutate state; the integrator owns updates — which is what makes multi-stage schemes trivial:

```
 Simulation::step(dt)
        │
        ▼
   Integrator ──────────────────────────────────────────────┐
        │                                                   │
        │  for each stage:                                  │
        │    Model::fill_ghosts(grid, state, t_stage)       │  halos + BCs
        │    Scheduler::for_each_block ──► vector fields    │  parallel, disjoint
        │                                                   │
        │  combine stages:  state.axpy_with(scheduler, …)   │  pure vector algebra
        │  stochastic term: state += Σⱼ √dt · Vⱼ(Y) ∘ ξⱼ    │  counter-based ξ
        └───────────────────────────────────────────────────┘
        │
        ▼
   Observers (async pipeline; never blocks the solver)
```

Because every state-shaped buffer (integrator stages, RHS accumulators, noise amplitudes) is **slab-congruent** with the state, integrators are grid-, dimension-, and scheme-agnostic vector algebra. RK4 is ~40 lines.

**Dynamics are driver-indexed vector fields.** A model is `dY = V₀(Y,t)·dt + Σⱼ Vⱼ(Y,t)·dWⱼ`: one `vector_field_block(driver, …)` evaluates the field conjugate to the time driver or to any of the model's independent Wiener drivers, and the model's driver set is a *type* (`NoNoise`, `Wiener<M>`). The √dt lives in the integrator, not the model; scaling noise by `dt` instead of `√dt` — a classic bug in hand-rolled stochastic solvers — is *inexpressible* here. So is pairing a deterministic-only scheme with a stochastic model: `RungeKutta4` implements `Integrator<G, D, NoNoise>` only, so that mistake is a compile error. Correlated components need nothing from the framework — drivers are independent by construction, and a model expresses correlation by mixing drivers across its fields (the Cholesky factor is model mathematics).

---

## Determinism guarantees

Reproducibility is a design constraint, not an aspiration:

- **Scheduling-independence.** Block writes are disjoint and land at fixed slab locations; the test suite asserts serial and Rayon runs are **bitwise identical** — including stochastic runs.
- **Counter-based noise.** There is no RNG stream to advance. Every random increment is a pure function of `(seed, step, driver, block, cell)` via a SplitMix64 chain + Box–Muller. Any worker, any thread count, any execution order: identical noise. Reproducing a run requires only its seed.
- **Fixed-order reductions** (planned, see roadmap) will extend the same guarantee to inner products for the implicit stack.

---

## Observation pipeline

The solver never blocks on IO. Snapshots move through a **pre-allocated ring** and a **bounded mpsc channel** to a background tokio runtime where sinks do the slow work:

```
 solver thread                        background thread (tokio)
 ─────────────                        ─────────────────────────
 observe(step, t, &state)
   ├─ free ring buffer? ──no──► skip (counted, never blocks)
   ├─ copy_from(state)                    ▼
   └─ try_send ────bounded mpsc───► ParquetSink ── snap_0042000.parquet
                                    FieldStatsSink ── progress bar stats
                 ◄────buffer return──────┘
```

Backpressure degrades *observation*, never *simulation*: if sinks fall behind, snapshots are dropped and counted. One Parquet file per snapshot means a killed run keeps every completed snapshot (a parquet file is valid only once its footer lands).

The on-disk format separates what changes from what doesn't — and is already shaped for AMR:

```text
run_dir/
├── static_<epoch>.parquet   x, y[, z] + static fields (θ₀, …), one per grid epoch
└── snap_<step>.parquet      step, t, epoch + dynamic fields, joined by row order
```

Coordinates and time-invariant fields are written once per **grid epoch** (a uniform run has exactly one; an AMR regrid bumps the epoch and emits a fresh static file — snapshots name their epoch in a ~free RLE column). Rows stream through **bounded row groups** (4M rows), so the writer's transient memory is flat in domain size instead of materializing multi-GB whole-domain columns that could evict the simulation's own state.

Live progress via `indicatif`:

```
⠁ [00:01:12] ████████████░░░░░░░░ 45231/187000 (628 steps/s, eta 3m 46s)
  t=723.7 | phi∈[-1.00,1.00] ⟨phi⟩=-0.53 frac>0: 23.4% | u∈[-0.70,0.09] ⟨u⟩=-0.512
```

---

## What is included

| Layer | Shipped today |
|---|---|
| **Geometry** | `CartesianGrid<const D>` (uniform, block-tiled, any dimension), signed ghost indexing, dimension-sweep halo exchange with mirror (no-flux) physical boundaries |
| **Discretization** | `FiniteDifference` (central, 2nd order), `FiniteVolume` (Karma–Rappel anisotropic flux divergence); operator tags `Laplacian`, `Gradient`, `Divergence`, `AnisotropicDivergence` |
| **Integrators** | `ForwardEuler`, `EulerMaruyama` (√dt-correct stochastic), `RungeKutta4` (O(dt⁴) drift, composable with noise) |
| **Models** | `ModelC` — Karma–Rappel dendritic solidification: coupled φ/u, 4-fold anisotropy, multi-grain nucleation with **per-grain crystallographic orientation** (static θ₀(x) field via nearest-seed Voronoi; anisotropy evaluated as A(θ − θ₀)); O(cells + blocks·grains) initialization |
| **Execution** | `SerialScheduler` (oracle), `RayonScheduler` (work-stealing), worker-pinned `ScratchPool`, scheduler-parallel state algebra |
| **Observation** | `AsyncObserver` pipeline, `ParquetObserver<D>` (long-format, snappy, x/y/z columns), `FieldStatsSink` + `indicatif` progress, movie rendering script (`scripts/render_model_c.py`) |
| **Stochastics** | counter-based, schedule-independent Gaussian field noise (`util::rng`) |

---

## Quick start

A complete model — the D-dimensional heat equation — lives in the crate-root docs and compiles as a doctest; the same pattern at full scale is `examples/heat3d.rs`. Defining a model means implementing `register_fields`, `fill_ghosts`, and `vector_field_block`; everything else (parallelism, stage buffers, output) is wiring:

```rust
let grid = CartesianGrid::new([96; 3], [32; 3], [0.0; 3], [0.1; 3])?;
let mut sim = Simulation::new(
    grid, FiniteDifference, Heat::<3> { kappa: 0.7, u: None },
    ForwardEuler, RayonScheduler, SystemAllocator,
);

let observer = AsyncObserver::new(
    200,                        // snapshot cadence (steps)
    sim.snapshot_buffers(3),    // pre-allocated ring
    vec![Box::new(parquet_sink), Box::new(stats_sink)],
);
sim.attach_observer(Box::new(observer));

let dt = sim.stable_dt().unwrap();
for _ in 0..steps { sim.step(dt); }
```

Run the shipped examples (`--help` lists every flag):

```bash
# classic single dendrite (630², corner seed), Parquet + progress + stats
cargo run --release --example model_c

# macroscale nucleation: 101.6M cells, 1000 grains, random orientations,
# stochastic sidebranching
RUSTFLAGS="-C target-cpu=native" cargo build --profile maxperf --example model_c
./target/maxperf/examples/model_c --cells 10080 --block 126 \
    --seeds 1000 --orient --noise 0.02 --time 350 --every 4000 --ring 2

# 3D heat diffusion through the same pipeline
cargo run --release --example heat3d
```

---

## Rendering results

Runs are rendered to video with the bundled Python script (numpy + pyarrow + matplotlib; ffmpeg on PATH for .mp4):

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt

.venv/bin/python scripts/render_model_c.py data/model_c --out dendrite.mp4
.venv/bin/python scripts/render_model_c.py data/model_c --field u      # thermal field
.venv/bin/python scripts/render_model_c.py data/model_c --grains       # color by θ₀
.venv/bin/python scripts/render_model_c.py data/heat3d --field u       # 3D mid-plane slice
```

The script reads the epoch-static + snapshot Parquet layout directly and reconstructs frames through the coordinate columns, so it is exact for any block decomposition and dimension (3D runs render the mid-plane slice).

---

## Validation

Every layer is pinned by an *exactness* test, not a tolerance hand-wave:

| Test | What it proves | Criterion |
|---|---|---|
| 2D/3D discrete eigenmode decay | grid, views, halo exchange, stencil, integrator, vector algebra — the whole chain | matches the analytic per-step amplification factor to `1e−10` over 200 steps |
| RK4 eigenmode decay | multi-stage plumbing incl. per-stage ghost refills | matches the quartic Taylor factor `1 − z + z²/2 − z³/6 + z⁴/24` exactly (RK4 is exact for linear modes) |
| Convergence orders | stage weights on a *nonlinear* problem | observed order ≈ 1 (Euler), ≈ 4 (RK4) under dt halving |
| serial ≡ parallel | scheduling-independence, incl. **stochastic** runs | bitwise equality |
| 3D heat conservation | mirror BCs + stencil are exactly conservative | Σu constant to all printed digits over 1050 steps |
| Solidification physics | the shipped phase-field model | φ stays in its wells, seed grows, latent heat bounded by the melting point |
| Halo exchange / mirror BCs | exact ghost values, every layer, cross-block and boundary | cell-exact assertions |
| Static-field layout | tendency buffers skip zero-tendency fields | zero-length slabs; axpy/noise leave statics bit-identical |
| Parquet round-trip | the on-disk snapshot contract | doubles round-trip bit-for-bit; row order matches the static file |
| Async pipeline | delivery, ring recycling, drain-on-shutdown | exact snapshot schedule received; `finish()` runs |
| Counter RNG | distribution + determinism | mean/variance of 2·10⁵ deviates; key sensitivity |

A numerical note recorded in the code: the textbook explicit thermal step `dt = 0.25 h²/D` sits **exactly** on the 2D stability limit, where the grid-scale checkerboard mode is undamped and secularly forced by latent-heat release — it eventually blows up. `legendre` uses `r = 0.2` (damping factor −0.6) and documents why.

---

## Performance

Measured on Apple Silicon (12 cores), release profile:

- **210² × 25,000 steps ≈ 7–8 s** (≈ 7 ns per cell-step, observation included) for the dendritic solidification model.
- **101.6M cells** (10080² as 80×80 blocks of 126²) runs at ~12 GB peak with Euler–Maruyama + a 2-deep snapshot ring; multi-seed initialization is O(cells + blocks·seeds) and takes seconds, not hours.
- No allocation after `Simulation::new`: field storage, stage buffers, worker-pinned scratch, and the snapshot ring are all created once.
- Blocks are the parallel unit; state algebra (`axpy`, copies, noise) is scheduler-dispatched too, since at large volume those memory-bound sweeps would otherwise dominate multi-stage integrators.

For maximum throughput: `--profile maxperf` (fat LTO, single codegen unit) + `RUSTFLAGS="-C target-cpu=native"`; PGO adds ~5–10% on top via `cargo-pgo`.

---

## Scope: what this framework can and cannot solve

Honesty section. The trait surface was audited against PDE classes beyond the ones implemented; here is the map.

### Current Solutions Scope

Any method-of-lines system **∂Y/∂t = F(Y, ∇Y, ∇²Y, …, x, t) + Σⱼ Vⱼ(Y)·ξⱼ** where F is evaluable cell-locally from a ghost-filled neighborhood:

- **Parabolic:** heat, diffusion, reaction–diffusion (Gray–Scott, FitzHugh–Nagumo, …)
- **Phase-field:** Allen–Cahn, dendritic solidification (shipped), Cahn–Hilliard (biharmonic = ghost width 2, already supported per-field)
- **Coupled multi-field systems** with cross-field RHS dependencies (the shipped model's `∂u/∂t` reads the freshly computed `∂φ/∂t` via split borrows)
- **Stochastic PDEs** with additive or multiplicative noise, bit-reproducibly
- **Hyperbolic** (advection, Burgers, wave as first-order system, shallow water): the *traits* are sufficient; what's needed are upwind/WENO/limiter stencil implementations — new `Stencil` impls, zero trait changes. Wide stencils are supported via per-field ghost widths.

### Expressible with Additive Extensions

| Class | Missing piece | Why it's additive |
|---|---|---|
| **Elliptic / implicit** (Poisson, steady states, stiff implicit stepping) | deterministic `State::dot`/`norm` reductions → matrix-free Krylov (CG/BiCGStab) → `ImplicitIntegrator` | a `Stencil` *is* already a linear operator `y = Ax`, and `State` is already the vector space Krylov needs; no existing trait changes |
| **Incompressible Navier–Stokes, quasi-static elasticity** | the implicit stack above (pressure projection / global solve) | models-as-bounds pattern unchanged |
| **IMEX splitting** for stiff reactions | optional stiff/non-stiff RHS split on `Model` | default method, existing models unaffected |
| **Adaptive CFL** (advection-dominated) | `stable_dt` currently cannot see the state; needs the same reductions | optional `stable_dt_state` method |
| **Periodic boundaries** | periodicity flags in `CartesianGrid::face_neighbor` | small, local |
| **AMR** | quadtree/octree grids behind the same `Grid` GAT-view contract | designed-for from day one: blocks are the refinement unit; stencils own coarse–fine interfaces |

### Outside the Current Architecture

- **Global / nonlocal operators** — spectral (FFT) methods, integral terms, boundary-integral formulations. `Discretizes::Stencil` deliberately binds operator realizations to block-local computation. (The serial pre-RHS phase can materialize nonlocal terms into auxiliary fields — see the `stencil` module docs — but a first-class `DiscretizesGlobal` trait deserves its own design round.)
- **Single-input stencils** — `Stencil::apply` takes one input field. Multi-input operators (the *oriented* anisotropic divergence reads φ and θ₀) currently use inherent methods on the concrete stencil, pinning the model to that stencil family via an associated-type bound. Generalizing the stencil arity is a known, contained extension.
- **Stretched / curvilinear grids** — `spacing` is per-block uniform (right for uniform grids and level-based AMR); per-cell metrics would extend the `Grid` surface.
- **Unstructured meshes / FEM**, **distributed MPI**, **GPU kernels** — explicit v1 non-goals. The seams they will eventually enter through (Scheduler for MPI/GPU dispatch, StorageBackend for device memory, halo exchange for rank boundaries) exist and are stable.

---

## Roadmap

In dependency order — each stage independently testable, none requiring redesign:

```
 deterministic reductions ──► matrix-free Krylov ──► implicit heat ──► projection
 (State::dot, fixed block     (CG/BiCGStab on           (validated       Navier–Stokes
  order; also unlocks          stencil-as-operator       against the
  adaptive CFL)                + ghost fills)             explicit one)

 in parallel:  periodic BCs · upwind/WENO stencil family · Cahn–Hilliard
 later:        multigrid on the block hierarchy · AMR grids · GPU storage backend
```

The design rationale lives where it can't rot: as doc comments on the traits themselves (`core/state.rs`, `geometry/grid.rs`, `discretization/operators.rs`, `physics/model.rs`, `core/scheduler.rs`, `core/observer.rs`). Start there.

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Minimum supported Rust version: **1.91**.
