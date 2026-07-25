# AMR for legendre: a design study

*Sources: M. Berger & J. Oliger, "Adaptive Mesh Refinement for Hyperbolic
Partial Differential Equations" (Stanford/Courant report, 1983; ADA130162 —
the tech-report form of the 1984 JCP paper), and M. Berger & I. Rigoutsos,
"An Algorithm for Point Clustering and Grid Generation" (IEEE Trans. SMC
21(5), 1991). These are the algorithms underlying Clawpack/AMRClaw.*

---

## 1. What the sources actually say

### 1.1 Berger–Oliger: the framework

**Hierarchy.** A fixed base grid G₀ covers the domain. Level-ℓ grids are
axis-uniform rectangular patches with spacing h_ℓ = h_{ℓ−1}/r (they prefer
r = 4; up to 10 in their shock-tube example). Patches at level ℓ **overlay**
the coarser levels — each patch is an independent computational entity with
its own solution storage, and the coarse solution continues to exist (and be
integrated) underneath. This independence is what makes every patch a
locally-uniform, vectorizable unit and the whole method "a method of domain
decomposition … well suited for multiprocessor architectures." It is
precisely legendre's block model.

**Proper nesting.** Every level-ℓ patch must lie in the *interior* of the
union of level-(ℓ−1) patches (boundary contact allowed only at the physical
boundary). Nesting is in the union, not in a single parent — a fine patch
may straddle several coarse patches.

**Error estimation (Richardson).** To flag cells, take the solution at time
t and (a) advance two steps with the scheme Q on (h, k), (b) advance one
step with the same scheme on (2h, 2k). Then

    τ(x) ≈ [Q²u − Q_{2h}u] / (2^{q+1} − 2)

estimates the local truncation error for a scheme of order q, **without
knowing the PDE or the scheme's error form** — the same user-supplied
integrator does the estimation. Flag where τ > ε. Near discontinuities the
estimate is not asymptotically valid but remains a good refinement
criterion (∝ solution jump). Cells under level-(ℓ+2) grids are always
flagged so the rebuilt level ℓ+1 preserves nesting.

**Regridding.** Every few steps (3–4 per level in their experiments; finer
levels regrid more often), rebuild levels **finest → coarsest** so nesting
constraints see the most accurate flags. Flagged cells get a **buffer
zone** of b cells before clustering: the buffer is what lets a moving
feature stay inside its patch between regrids (regrid interval and buffer
width trade off directly). New fine patches are initialized by
interpolation from the finest existing data beneath them — except at t = 0,
where levels are built recursively from the *initial conditions themselves*
until τ < ε everywhere.

**Time refinement (subcycling).** k_ℓ = k_{ℓ−1}/r keeps λ = k/h constant
across levels; the integration is a recursion —

    Integrate(ℓ):
      repeat r times:
        (regrid time?) estimate error, rebuild ≥ ℓ
        step Δt_ℓ on all level-ℓ patches
        if level ℓ+1 exists: Integrate(ℓ+1); Update(ℓ, ℓ+1)

so a coarse step is the unit of work and the finest grid's small dt is
never imposed on the whole domain. (For parabolic dt ∝ h² the analogous
ratio is r² steps per level.)

**Intergrid operations.** Three, all of them topology:
1. *Fine-boundary values*: from an abutting same-level patch where one
   exists; otherwise interpolated in space (and, under subcycling, in time)
   from the coarse level. Stability of the interface schemes is provable
   (Berger 1982).
2. *Updating (restriction)*: when levels are synchronized, inject the fine
   solution onto underlying coarse cells. Not cosmetic — without it the
   coarse solution disperses under the patch and contaminates the very
   values later used for fine-boundary interpolation.
3. *Same-level overlap averaging* — irrelevant if same-level patches are
   kept disjoint, which we will (modern practice; B–R's rectangles are
   disjoint by construction).

**Cost.** Measured overhead (estimation + regridding + interpolation)
≈ 12% of runtime; equal-accuracy speedups of 4–7× over uniform fine grids,
and access to resolutions where the uniform run is simply infeasible.

**What we deliberately drop from 1983:** rotated rectangles (their ellipse
moment-fitting orientation step) and the nearest-neighbor / spanning-tree
clustering. Both were superseded by —

### 1.2 Berger–Rigoutsos: the clustering algorithm

Flagged cells on one level form a binary image. For a candidate box R with
per-axis **signatures** Σ_d (counts of flagged cells in each plane
perpendicular to axis d):

```
worklist = [bounding box of all flags]
while let Some(R) = worklist.pop():
    if efficiency(R) = flags/cells ≥ η  →  accept R
    else:
        1. HOLES: if any signature has a zero entry, split at the hole
           nearest the box middle (separates islands; always exploited
           before inflections)
        2. INFLECTION: else compute Δ_d = second difference of Σ_d; find
           the zero crossing of Δ with the largest |change|, across all
           axes; split there
        3. FALLBACK: else bisect the longest axis (the paper's fix for
           its own 45°-stripe anomaly, which otherwise floors at 50%)
        push both halves (each shrunk to its flags' bounding box)
```

Cost O(k·(P + ΣN_d)) for k output boxes and P flags; achieved efficiency
typically 85–100%. Dimension-agnostic: signatures are 1-D arrays per axis,
so the same code is 2-D and 3-D (and D-dimensional). Knobs: efficiency
threshold η (0.7–0.85 typical), minimum box size, and a max-boxes guard.

---

## 2. What legendre already has

The crate was built with these seams (deliberately, per its own docs):

| B–O requirement | legendre today |
|---|---|
| independent locally-uniform patches with own storage | blocks: `BlockStorage` per block, per-block `block_len` (variable lengths already supported) |
| per-patch level/spacing | `Grid::level(block)`, `Grid::spacing(block)` — trait-level since day one |
| patch-parallel integration | `Scheduler::for_each_block_mut` — patches are just work items |
| geometry read at apply time, "one stencil serves all levels" | `Stencil::apply(grid, block, …)` reads `spacing(block)`; explicit AMR note in `stencil.rs` docs |
| regrid = new grid + state migration | pinned in `grid.rs` docs ("an AMR regrid produces a *new* grid… which keeps `Grid: Sync` trivially sound") |
| per-epoch output | Parquet observer's `static_<epoch>` / `snap_<step>` contract |
| deterministic noise on any grid | counter keys `(seed, step, driver, block, cell)`; well-defined per patch (streams re-key at regrid — documented behavior, same class as the driver re-keying) |
| initial conditions per level at t=0 | `fill_from_fn` evaluates f(cell center) — works verbatim on any level's patches |

The four genuinely missing pieces: cross-level ghost machinery,
`Discretizes` impls for a second grid family, a regrid orchestration layer
(a second allocating path), and a tagging API.

---

## 3. Proposed design

### 3.1 `AmrGrid<const D: usize>` (module `geometry::amr`)

**Refinement is strictly rectilinear**: patches are axis-aligned boxes of
uniform cells at each level — never a general placement of points, and (per
modern practice) never the 1983 paper's rotated rectangles. B–R is explicit
that "for numerical reasons the rectangles are oriented with the base
grid." This is the property that lets every existing stencil kernel run on
a patch unmodified.

A **flat forest of axis-aligned patches**, rebuilt wholesale at each regrid
(immutable between regrids — the existing `Grid: Sync` argument carries):

```rust
pub struct Patch<const D: usize> {
    level: u8,
    origin: [i64; D],   // global index at this patch's level
    extent: [usize; D], // interior cells
}

pub struct AmrGrid<const D: usize> {
    base: CartesianGrid<D>,        // level 0 geometry + periodicity
    ratios: Vec<u8>,               // r per level transition (B–O: prefer 4)
    patches: Vec<Patch<D>>,        // level-major order; BlockId = index
    // Precomputed at construction (see 3.3):
    exchange: ExchangePlan,        // same-level halo copies
    prolong: ProlongPlan,          // coarse→fine ghost interpolation
    restrict: RestrictPlan,        // fine→coarse interior averaging
}
```

- `Grid` impl: `View/ViewMut = CartesianView{,Mut}` — **the same view types
  as `CartesianGrid`**, since a patch is a uniform box. Stencil kernels are
  therefore reusable as-is; only the impl surface needs widening (3.2).
- `cell_key`: the existing per-patch interior linearization is correct
  unchanged (ids are per-block).
- Nesting, disjointness of same-level patches, and minimum patch sizes are
  **enforced at construction** (debug-assert + construction-time check);
  the B–R output is post-processed to satisfy them, so invariants hold by
  construction everywhere else.
- Level 0 is exactly the base `CartesianGrid`'s blocks: a 1-level `AmrGrid`
  is the uniform case with zero behavioral difference.

### 3.2 One trait unlock: `Rectilinear<const D: usize>`

The only structural change to existing code. Today's stencils are
implemented per concrete grid (`Discretizes<CartesianGrid<2>, …>`).
Introduce:

```rust
/// A grid family whose blocks are uniform axis-aligned boxes with
/// CartesianView views: everything a rectilinear stencil kernel needs.
pub trait Rectilinear<const D: usize>:
    Grid<Point = [f64; D], Index = [isize; D]>
    + for<'a> GridViews<'a, D>   // associated-type-equality to CartesianView
{
    fn block_extent(&self, block: BlockId) -> [usize; D];
}
```

and rewrite the existing `Discretizes`/`Stencil` impls **once** against
`G: Rectilinear<D>` instead of `CartesianGrid<D>`. The kernels don't change
a character (they already take `(grid, block, view)`); monomorphization
makes this zero-cost; and both grid families — plus any future one with
uniform boxes — get every operator. This mirrors the driver-consolidation
lesson: one vocabulary, no per-family duplication. (If the GAT-equality
bound proves too noisy in practice, the fallback is a small
`impl_rectilinear_stencils!(GridType)` macro — same kernels, uglier
plumbing; the trait is preferred.)

### 3.3 Ghost machinery as precomputed plans (module `geometry::amr`)

All three B–O intergrid operations are **pure topology**, computed once per
regrid and executed per step as flat copy/interpolate lists (no searching
in the hot loop, scheduler-dispatchable, deterministic by construction):

- `ExchangePlan`: `(dst patch, dst ghost box, src patch, src box)` for
  same-level abutting patches — the analogue of `fill_ghosts_mirror`'s
  neighbor arm, including periodic wrap via the base grid's topology.
- `ProlongPlan`: for ghost regions with no same-level neighbor, the
  coarse-level source boxes + interpolation stencil (v1: bilinear/trilinear
  in space; time interpolation only arrives with subcycling). B–O's
  stability analysis is the license for interpolated internal boundaries.
- `RestrictPlan`: after each synchronized step, conservative averaging of
  each fine patch's interior onto the underlying coarse cells (r^D-cell
  means) — B–O's "updating," which is mandatory, not cosmetic.

Model-facing API stays one line, in the same shape as today:
`amr::fill_ghosts(grid, state, handle, boundary_rule)` — same-level
exchange + prolongation + the physical rule (mirror/extrapolate closures,
shared with the Cartesian helpers) at true domain boundaries.

### 3.4 Tagging: a driver-level trait, not a `Model` method

Applying the `has_noise` lesson: refinement criteria don't belong on
`Model` (bound-strengthening trap, and most models will never refine).

```rust
pub trait TagCells<G, D>: Send + Sync {
    /// Mark cells of `block` needing refinement (write 1.0 into `tags`).
    fn tag_block<S>(&self, ctx: &RhsContext<'_, G, D>, state: &State<f64, S>,
                    tags: &mut BlockStateMut<'_, f64, S>, …);
}
```

Shipped implementations:
- `GradientTagger { field, threshold }` — |∇u| h > ε, the pragmatic v1
  (Model C: tag the interface via |∇φ|).
- `RichardsonTagger<I>` (v2) — B–O's estimator, built from the *existing*
  integrator machinery: coarsened step vs. two fine steps on scratch
  buffers, scheme-agnostic exactly as in the paper.

### 3.5 Adaptivity as a policy on the existing `Simulation` — no second driver

The same consolidation lesson as the driver refactor: a parallel
`AmrSimulation` would duplicate the ownership graph and split the
ecosystem. Instead, `Simulation` gains one defaulted policy parameter:

```rust
/// Decides whether and how to rebuild the grid. All AMR knowledge —
/// tagging, clustering, nesting, plan construction, state migration —
/// lives in implementations of this trait, none in Simulation.
pub trait Adapt<G: Grid, T: Scalar, S: StorageBackend<T>, A>: Send + Sync {
    /// Called once per step. `Some((new_grid, migrated_state))` means
    /// regrid: the policy has built the new hierarchy and migrated the
    /// solution onto it (same-level overlap copies + prolongation of
    /// newly refined regions). `None` means keep stepping.
    fn regrid(&mut self, grid: &G, state: &State<T, S>, alloc: &A, step: u64)
        -> Option<(G, State<T, S>)>;
}

/// The default: never adapts. `regrid` returns `None` and the call
/// monomorphizes to nothing — uniform-grid simulations pay zero cost and
/// keep every existing signature via the default type parameter.
pub struct Static;
```

`Simulation<G, D, M, I, Sch, A, R = Static>`: `step()` consults `R` first;
on `Some` it swaps grid and state and rebuilds stage buffers and scratch —
machinery it already owns (`stage_layout` → `like_for`/`like`) — the
documented *second allocating path*, amortized over the regrid interval
(B–O measured ~12% total overhead). Existing constructors are untouched;
`Simulation::adaptive(…, policy)` is the one addition. Returning the
*migrated state* from the policy is the move that keeps `Simulation`
AMR-ignorant: no `G: Regriddable` bounds, no migration trait on `Grid`.

The shipped policy is `BergerOliger<Tag: TagCells>`, owning the
`RegridPolicy { every, buffer, efficiency, min_box }` knobs and the v1
(global-dt) pipeline: tag finest→coarsest with buffer dilation and
level-(ℓ+2) nesting flags → B–R cluster per level → nesting/disjointness
post-pass → new `AmrGrid` + plans → migrate. Between regrids, `step()` is
the existing integrator over all patches at once, plus fine→coarse
restriction after the update.

```rust
let grid = AmrGrid::base(CartesianGrid::new(…)?).ratios(&[4, 4]).build();
let mut sim = Simulation::adaptive(grid, disc, model, integrator, scheduler,
    SystemAllocator,
    BergerOliger::new(
        GradientTagger { field: phi, threshold: 0.1 },
        RegridPolicy { every: 4, buffer: 2, efficiency: 0.8, min_box: 8 }));
for _ in 0..steps { sim.step(dt); }
```

**One honest gap this surfaces:** observers capture the grid at
construction (`ParquetObserver::new(grid.clone(), …)`). Under regrids the
`Observer` trait needs the current grid — either `observe` gains a grid
argument or an `on_regrid(&G)` hook bumps the observer's epoch. Mechanical
and small; the parquet format's `static_<epoch>` mechanism was designed for
exactly this event.

**v2 (Berger–Oliger subcycling)** — the recursive per-level cycle with
two-time-level coarse boundary storage for time interpolation — is a
different integrator orchestration layered on the same policy machinery.
It is where the hyperbolic-workload payoff lives; v1's global dt is correct
for everything and simplest for the parabolic workloads legendre runs
today.

**Refluxing** (conservation fix-up at coarse-fine faces, Berger–Colella
1989) is out of v1 scope, documented: phase-field/SDE workloads don't
require it; conservative finite-volume shock problems eventually will.

### 3.6 I/O and visualization

Two sinks, both epoch-aware, serving different consumers:

**Parquet (ours, source of truth).** The existing epoch mechanism already
models regrids: on every regrid the observer's epoch bumps and a new
`static_<epoch>` is written. AMR needs only extra columns there — per cell:
`patch` (BlockId), `level`; per patch: origin, extent, spacing. Because the
render scripts already draw *by coordinates* (exact for any block layout),
existing quick-look tooling keeps working with two additions in Python:
patch outlines (one `matplotlib` `Rectangle` per patch from the static
table) and **finest-wins compositing** (mask any cell whose center lies
inside a finer patch's box — computable per epoch from the patch table
alone). This is the cheap, always-on path.

**AMReX/BoxLib plotfile (industry standard, feature-gated).** The de facto
interchange for block-structured AMR, natively read by **yt**, **ParaView**
(AMReX reader), and **VisIt** (Boxlib reader). Crucially, the format is an
ASCII `Header` (levels, boxes, times, ratios) plus per-level raw FAB
binaries — **writable in pure Rust with zero native dependencies**, unlike
every HDF5-based alternative. A `PlotfileObserver` (feature `amrex-io`)
gives the canonical Python AMR visualization for free:

```python
import yt
ds = yt.load("plt00040")
p = yt.SlicePlot(ds, "z", ("boxlib", "phi"))
p.annotate_grids()   # draws the patch hierarchy per level
p.save()
```

**Considered and rejected:** VTK HDF `OverlappingAMR` (the modern
ParaView-native AMR format) and Chombo HDF5 — both excellent, both
requiring `libhdf5` bindings, which violates the crate's dependency
posture. Revisit only if a consumer demands them; the plotfile route
already covers ParaView and VisIt.

### 3.7 Ergonomics, determinism, performance

- **Enablement is a type swap plus one argument**: `CartesianGrid` →
  `AmrGrid::base(…)`, `Simulation::new` → `Simulation::adaptive(…, policy)`.
  Models, integrators, schedulers: unchanged (models via the `Rectilinear`
  unlock); observers gain one epoch hook (3.5). No feature flag needed —
  unused AMR code is dead-stripped; a 1-level `AmrGrid` *is* the uniform
  grid and `R = Static` *is* today's simulation.
- **Determinism**: tags are pure functions of state; B–R is deterministic;
  plans are data. Serial ≡ parallel bitwise is preserved (all sweeps remain
  disjoint block writes). Stochastic streams re-key at regrids (block ids
  change) — same documented class as the driver re-keying.
- **Performance**: no per-cell branching anywhere new — plans are flat
  lists executed like halo copies; prolongation cost scales with coarse-
  fine *surface*, not volume; clustering is O(patches·flags) every K steps.
  Flamegraph evidence says legendre's cost is stencil-dominated, so AMR's
  win is exactly B–O's: don't run stencils where nothing is happening
  (their 12%-refined rotating-cone run cost 16% of the uniform fine run).

---

## 4. Phasing (each lands green, none blocks the market-maker migration)

| Phase | Content | Validation |
|---|---|---|
| **A** | `amr::cluster`: Berger–Rigoutsos in D dimensions as a pure function `(&flags-boxes, η, min_box) → Vec<Box>` | unit tests reproducing the paper's cases: islands (holes), V/L-shapes (inflections), 45° stripes (bisection fallback), efficiency assertions |
| **B** | `Rectilinear` trait + stencil-impl migration; `Patch`/`AmrGrid` with `Grid` impl and nesting enforcement | existing suites unchanged (uniform grids через the trait, bit-identical); patch geometry unit tests |
| **C** | plans (exchange/prolong/restrict) + `amr::fill_ghosts` on a **static** two-level hierarchy; parquet patch columns + render-script outlines/compositing | heat eigenmode with a hand-placed fine patch: interface consistency, restriction conservation, serial ≡ rayon bitwise; rendered hierarchy inspected |
| **D** | `Adapt` policy param on `Simulation` (+ observer epoch hook), `TagCells` + `GradientTagger`, `BergerOliger` with regrid/migration, global dt; `PlotfileObserver` (feature `amrex-io`) | Model C dendrite with |∇φ| tagging vs. uniform-fine reference (the B–O comparison methodology: equal accuracy, fraction of cost); `yt` loads the plotfiles; CodSpeed bench `amr/step` |
| **E** | subcycling driver, `RichardsonTagger`, refluxing | B–O shock-tube-style convergence + conservation tests |

## 5. Open questions (decide during Phase B/C, flagged now)

1. **`Rectilinear` bound mechanics** — GAT equality vs. macro fallback;
   prototype decides.
2. **KarmaRappelFlux at coarse-fine corners** — the two-input oriented
   stencil reads θ₀ and φ; prolongation of static fields (θ₀) happens at
   regrid only, but corner ghost interpolation order needs care (bilinear
   may be insufficient for the anisotropy's fourth derivatives; B–O used
   quadratic interpolation for second-order interior schemes).
3. **Buffer/regrid-interval defaults** — B–O: buffer 1–4, regrid every 3–8
   steps; tune per workload via `RegridPolicy`, benchmark on Model C.
4. **Patch size floor vs. scheduler granularity** — B–R can emit small
   boxes; `min_box` + a merge pass keeps work items worth dispatching.
