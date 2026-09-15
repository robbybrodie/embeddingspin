# Patent Alignment

Application **20251253US**, *Hierarchical Phase-Encoded Temporal Vectors for Semantic
Retrieval* — status: **pending**.

This document maps the application onto the code: every claim to the function that
implements it and the test that pins it, then the features the specification
discloses beyond the claim set. It is the reference to reach for when asking "is the
repository still doing what the application says it does?"

Run the conformance suite on its own with:

```bash
./venv/bin/python -m pytest tests/test_patent_conformance.py
```

---

## 1. Notation

Formula 1, as implemented:

```
φ = 2π · fmod_floored( (t − t₀) / T , 1.0 )
```

`t` is a moment, `t₀` the base epoch, `T` the period of the circle in years. The
modulo is applied to the **period ratio**, before multiplication by 2π, so the
operation stays dimensionless — `floored_mod(value, modulus=1.0)` in
`temporal_config.py:158`.

Each scale contributes a triple:

```
[cos φ, sin φ, z]
```

`φ` is the phase of the **midpoint** of the interval and `z` its arc length in
radians. `z = 0` is a point; `z > 0` is an arc, capped at 2π. **N** scales give a
**3N**-dimensional temporal vector appended to the frozen semantic embedding.

## 1.1 The claims recite N scales, not three

This matters for how the repository is read, so it is worth stating plainly.

Independent claims 1, 15 and 20 recite **a first** and **a second** periodic scale —
two, as the floor. Claim 2 adds a third. Nothing in the claim set caps the count, and
claim 4 ("integer powers of two") and claim 7 ("a factor of the second") constrain
the *relationship* between periods rather than how many there are.

The implementation matches that scope. `TemporalHierarchy` holds an ordered,
variable-length tuple of `ScaleSpec` from finest to coarsest; the count is validated
as ≥ 1 and is otherwise unbounded. Nothing downstream hardcodes three or nine:

| Concern | How it generalises |
|---|---|
| vector width | `dimensions = 3 × len(scales)` |
| reading a stored vector | strips `3 × tuple_count` from the header, never a constant |
| splitting | at `hierarchy.finest`, whichever scale that is |
| traversal | `reversed(hierarchy.scales)`, coarsest first |
| weighting | `normalized_weights()` over the scales actually traversed |
| coverage | epoch + `hierarchy.coarsest.period_years` |

Verified across configurations:

| Scales | Dims | Coverage | Traversal for a quarter query |
|---|---|---|---|
| 2 (1y, 16y) | 6 | 1916 | `decade`, `quarter` |
| **3 (default)** | **9** | **2156** | `century`, `decade`, `quarter` |
| 4 (+ 4096y) | 12 | 5996 | `millennium`, `century`, `decade`, `quarter` |
| 5 (+ 65536y) | 15 | 67436 | `eon`, `millennium`, `century`, `decade`, `quarter` |

The three-circle hierarchy is the **shipped default**, and the one the published
worked example and the test suite's `== 9` assertions pin. It is a configuration, not
the method. Where this document says "three circles" it is describing
`DEFAULT_HIERARCHY`.

---

## 2. Claim-by-claim map

### Independent claim 1 — encoding a semantic vector

| Element | Implementation | Test |
|---|---|---|
| accessing a semantic vector from a semantic embedding model | `ingestion.TemporalSpinIngestionPipeline.ingest_document` → `llamastack_client` / `openai_client` | `tests/test_retrieval.py::TestIngestion` |
| determining temporal information | `TemporalSpinIngestionPipeline.resolve_interval`, `temporal_spin.extract_interval_from_text` | `tests/test_retrieval.py::TestIngestion` |
| mapping to a first phase on a first periodic scale | `temporal_encoding.phase_of` (line 226) | `test_patent_conformance.py::TestFormula1` |
| mapping to a second phase on a second periodic scale | same, per `ScaleSpec` in `TemporalHierarchy.scales` | `test_patent_conformance.py::TestVectorStructure::test_periods_are_one_sixteen_and_two_hundred_fifty_six_years` |
| generating a temporal vector | `temporal_encoding.encode_tuple` → `TemporalEncoding.to_vector` | `test_temporal_encoding.py::TestEncodeTuple` |
| storing in association with the semantic vector | `vector_store.VectorStore.add_documents`, `SpinDocument.full_embedding` | `test_vector_store.py::TestRoundTrip` |

### Dependent claims

| # | Claim | Implementation | Test |
|---|---|---|---|
| 2 | third periodic scale | `CENTURY_SCALE` (256y) in `DEFAULT_SCALES`; `MILLENNIUM_SCALE` (4096y) available | `TestVectorStructure::test_three_circles_give_nine_dimensions` |
| 3 | Z-value as arc length of an interval | `ScaleTuple.z`, computed in `encode_tuple` | `TestVectorStructure::test_z_is_zero_for_a_point_and_positive_for_a_duration`, `…::test_an_arc_is_capped_at_one_full_revolution` |
| 4 | periods are integer powers of two | 1 = 2⁰, 16 = 2⁴, 256 = 2⁸, 4096 = 2¹² | `test_temporal_config.py::TestScaleSpec` (powers of two) |
| 5 | first period one year, second sixteen | `QUARTER_SCALE`, `DECADE_SCALE` | `TestVectorStructure::test_periods_are_one_sixteen_and_two_hundred_fifty_six_years` |
| 6 | phase carried as a sine and a cosine | `ScaleTuple.cos` / `.sin`, emitted by `as_triple()` | `TestQ1_2026WorkedExample::test_published_tuple_values` |
| 7 | first period is a factor of the second | validated in `TemporalHierarchy.__post_init__` | `test_temporal_config.py::TestScaleSpec` (each period divides the next) |
| 8 | floored modulo, remainder never negative | `temporal_config.floored_mod` | `TestFormula1::test_the_remainder_is_non_negative_and_below_the_modulus` |
| 9 | scales are model-agnostic | no model reference anywhere in `temporal_config` / `temporal_encoding`; the encoder takes a `datetime`, never an embedding | `test_retrieval.py` runs the whole stack on `MockEmbeddingClient` |
| 10 | temporal vector appended to the semantic vector | `SpinDocument.full_embedding = semantic + temporal`; readers strip `3 × tuple_count` trailing dimensions | `test_vector_store.py::TestRoundTrip`, `…::TestHeaderDrivenSplit` |
| 11 | semantic embedding of text | `ingest_document(text=…)` throughout | `test_retrieval.py::TestIngestion` |
| 12 | temporal information is an instant | `TemporalInterval.point`, `z = 0` at every scale | `test_temporal_encoding.py::TestTemporalInterval` |
| 13 | a range of time reduced to an instant | `TemporalInterval.midpoint` — `cos`/`sin` encode the midpoint, `z` the extent | `TestQ1_2026WorkedExample::test_the_midpoint_is_february_fifteenth` |
| 14 | the scales are circular | phases live on `[0, 2π)`; wraparound handled by `angular_difference`, `arc_overlap`, `_linear_spans` | `test_temporal_encoding.py::TestArcAlgebra` |

### Independent claim 15 — querying

| Element | Implementation | Test |
|---|---|---|
| accessing a query vector | `retrieval.TemporalSpinRetriever.create_query` | `test_retrieval.py` |
| determining temporal information for the query | `create_query(interval=…)`; absent ⇒ `full_span_interval()` | `test_retrieval.py::TestUnconstrainedQuery` |
| mapping to phases on two periodic scales | `encode_single` over the same hierarchy as ingestion | `test_retrieval.py::TestRetrieverHierarchyGuard` |
| generating a query temporal vector | `SpinQuery.full_embedding` with `lambda_factor = 0.1` | `test_retrieval.py::TestOverlapGate` |
| accessing a corpus | `VectorStore.search` | `test_vector_store.py::TestSearchAndFilter` |
| selecting a subset | `TemporalSpinRetriever.search` | `test_retrieval.py::TestResults` |

| # | Claim | Implementation | Test |
|---|---|---|---|
| 16 | two-pass selection: semantic candidates, then temporal | pass 1 `top_k_coarse=200` on the concatenated vector; pass 2 `score_candidate` | `test_retrieval.py::TestOverlapGate` |
| 17 | second pass keeps only vectors overlapping at each scale | hard gate in `score_candidate`: any non-overlapping traversed scale returns `None` | `test_retrieval.py::TestOverlapGate::test_a_quarterly_query_excludes_sibling_quarters`, `test_patent_conformance.py::TestRetrievalSemantics::test_the_same_quarter_in_two_years_is_separated_by_the_coarser_circle` |
| 18 | the interval at a scale is a range | arc-to-arc branch of `evaluate_scale` | `test_temporal_encoding.py::TestEvaluateScale` |
| 19 | the interval at a scale is an instant | point branches of `evaluate_scale` (`arc_contains_point`) | `test_temporal_encoding.py::TestEvaluateScale` |

### Claim 20 — computing system

The FastAPI service in `api.py` is the system embodiment: it holds the hierarchy,
the embedding client and the store, and exposes encode-and-store (`/ingest`) and
query (`/temporal_search`) over HTTP. Covered by `tests/test_api.py` end to end
against the in-memory store.

---

## 3. Disclosed in the specification, beyond the claims

These are described in the application but not recited in claims 1–20. They are
implemented and tested all the same.

### 3.1 Configurable base epoch

`t₀` defaults to **1900-01-01 UTC**, giving coverage through **2156** (epoch + the
256-year circle). It is set by `EMBEDDINGSPIN_EPOCH` and read once, at import, by
`temporal_config.default_epoch()`.

The published worked example was computed against a 2010 epoch. That hierarchy is
kept as `PATENT_EXAMPLE_HIERARCHY` (coverage 2266) so the numbers in the application
remain reproducible:

```
Q1 2026, epoch 2010, midpoint 15 Feb 2026, 45 elapsed days
  quarter (1y)    cos 0.7147  sin 0.6995  z 1.5493
  decade  (16y)   cos 0.9988  sin 0.0484  z 0.0968
  century (256y)  cos 0.9227  sin 0.3855  z 0.0061
```

`TestQ1_2026WorkedExample` reproduces all three to 5e-5.

**Epoch changes are not free.** A stored phase survives a change of `t₀` only if the
shift is a whole multiple of *every* period. 2010 → 1900 is 110 years: a multiple of
1 but not of 16 or 256, so it requires a re-index. `epoch_shift_is_congruent` and
`describe_epoch_migration` (`temporal_config.py:556`, `:586`) answer the question
before you move; `TemporalHierarchy.fingerprint()` and the store's hierarchy guard
stop a mismatched corpus being queried by accident.

### 3.2 Calendar year convention

Position within the 1-year circle is a fraction of the **actual** calendar year, 365
or 366 days, so 1 January is phase 0 in every year including leap years
(`calendar_year_position`). The older fixed-length convention (365.2425 days) is
retained as `linear` for backward compatibility, selected by
`EMBEDDINGSPIN_YEAR_CONVENTION`.

### 3.3 Segments — even geometry, calendar identity

Each circle carries a segment count: the 1-year circle four calendar quarters, the
16-year circle sixteen calendar years, the 256-year circle sixteen 16-year blocks.
**Every circle is divided evenly**, and that is a hard constraint, not a
simplification. Segment counts are powers of two alongside the periods, so a segment
index is

```
index = int(fraction * segments)
```

on a fraction Formula 1 has already reduced to `[0, 1)` — one multiply and a
truncation. No divider table, no search, no data-dependent branch, and the same
arithmetic vectorises across a batch of candidates. That property is what the
at-scale performance argument rests on, so nothing is permitted to make the geometry
uneven.

**The calendar is uneven, and the correction is applied after the geometry, not
inside it.** Quarters are 90, 91, 92 and 92 days. 29 February sits inside Q1, so in a
leap year every later divider shifts one ordinal day out; and because
`calendar_year_position` measures each year against its own length (§3.2), that is a
different *phase*, not the same one — 1 April is `91/366 = 0.24863` in a leap year
against `90/365 = 0.24658` in a common one, a gap of 18 hours at the Q2 divider, 12
at Q3, 6 at Q4. No single fixed fraction can serve both, and a second divider table
selected per year would turn the index into a search over a non-uniform table: a
branch per candidate per scale.

So segment **identity** is not derived from a phase at all. It is resolved from the
real dates, once, at encoding time — the last point at which the dates are still in
hand — by `calendar_segment` and `calendar_segments_touched`
(`temporal_encoding.py`), in integer arithmetic:

```
quarter circle  : (month - 1) // 3
16-year circle  : (year - epoch_year) % 16
256-year circle : ((year - epoch_year) // 16) % 16
```

Exact by construction. There is no divider to approximate, so the uneven quarters and
the leap day need no special case, no second table and no tolerance. The result is
stored in the tuple, read back from the header, and never re-derived downstream.

The split of responsibility:

| Stage | What it holds | How segments are resolved |
|---|---|---|
| encoding (`encode_tuple`) | the interval, so the real dates | exactly, by integer calendar arithmetic; stored on the tuple |
| comparison (`evaluate_scale`) | phases only | the even windows, `int(fraction * segments)` |

The two deliberately disagree near a divider — 1 April is geometrically Q1 and
calendrically Q2 — and the disagreement is left alone. It cannot create a false
positive: the gate is the arc intersection, not the segment, and a per-segment score
can only raise a result that already overlaps. Measured across every quarter and
month pair of 2023 and 2024, an even window changes no gate decision and the worst
Jaccard delta is 0.

The coarse circles are exact in both spaces regardless: they divide into whole
numbers of years, and the year *count* is an integer, so the even divider and the
calendar divider are the same place.

An earlier revision derived identity from a phase against a table of calendar
dividers. Its residual error was confined to segment **identity** — the label
reported for an instant in the last hours of a leap-year quarter, 36 hours per leap
year — and never reached the overlap gate, which is geometric and was correct
throughout. It mattered because `api.py` surfaces `segment_label` to callers, and it
is resolved by construction now rather than by carve-out.

`tests/test_segments.py` sweeps every hour of eight sample years from 1900 to 2155,
and every month and quarter of every year from 1900 to 2156, on all three circles.

Segments also drive scoring. `evaluate_scale` computes a Jaccard coefficient
restricted to each shared segment and takes

```
jaccard = max( whole_arc_jaccard , max over shared segments )
```

so a query window straddling a divider — running from inside 2022 into 2023 — still
matches genuinely relevant documents on the far side, instead of losing them because
the query's centre of mass sat on the near side.

### 3.4 Zero-degree boundary splitting

An interval crossing a divider on the **finest** circle is stored as several
representations, one per period it spans, all sharing a `group_id`. A 2017–2022
review becomes six rows, each with a full 1-year arc (`z = 2π`) and a distinct phase
on the 16-year circle. Retrieval collapses them with `deduplicate_by_group`, so the
consumer sees the document once no matter which arc matched.

Splitting at the finest scale alone is sufficient precisely because each period is a
factor of the next (claim 7): a boundary on a finer circle is always also a boundary
on a coarser one. `MAX_REPRESENTATIONS` caps the fan-out; beyond it the interval is
kept whole.

`split_at_boundaries`, `boundary_moments` — `temporal_encoding.py:730`, `:704`.

### 3.5 Self-describing variable-length vectors

The application describes extending the hierarchy without re-indexing. That is only
true under stated conditions, and the code states them.

Every encoding carries a header — `tuple_count`, epoch, year convention, and per
tuple the period, segment count and weight. A reader strips exactly
`3 × tuple_count` trailing dimensions rather than assuming nine, so a corpus
containing both three-circle and four-circle vectors reads correctly
(`TestHeaderDrivenSplit`). Vectors written before an outer circle was appended are
brought forward with a **neutral tuple** `[cos = 1, sin = 0, z = 2π]` — a saturated
arc that overlaps everything and so cannot reject (`pad_to_hierarchy`,
`neutral_tuple`).

Appending an **outer** scale preserves every existing phase, because a finer circle's
phase does not depend on the periods above it. Changing the **epoch** does not
(§3.1). Both halves are tested: `TestExtensibility::test_appending_an_outer_scale_preserves_every_existing_phase`
and `…::test_extension_is_not_free_if_the_epoch_moves`.

### 3.6 Lazy resolution

A query descends only as far as it needs. A scale is skipped when the query arc
**saturates** that circle, because a full-circle arc overlaps everything and the
check cannot reject anything. "All of 2021" spans the whole 1-year circle, so the
1-year check is dropped and the year is resolved on the 16-year circle instead. The
plan is ordered **coarsest first**, so the widest and cheapest rejections happen
before any fine-grained work (`traversal_plan`, `TraversalPlan`).

| Query | Traversed |
|---|---|
| a single day | `century`, `decade`, `quarter` |
| a month or a quarter | `century`, `decade`, `quarter` |
| a full year | `century`, `decade` |
| 2017–2022 | `century`, `decade` |
| 2000–2050 (saturates the 16-year circle too) | `century` |
| unconstrained | `century` |

Encoding precision and traversal depth are separate concerns. Precision is fixed at
ingestion: every scale is always written, nothing is discarded. Depth is a property
of the question. Retrieval cost therefore tracks the precision the query asked for,
not the precision the corpus holds
(`TestLazyResolution::test_encoding_precision_and_traversal_depth_are_separate`).

### 3.7 Two-pass retrieval and β

Pass 1 is a coarse recall of 200 candidates over the concatenated vector with the
temporal block scaled by λ = 0.1. Pass 2 applies the hard overlap gate and blends:

```
temporal = Σ  w_s · jaccard_s          (weights normalised over traversed scales)
score    = (1 − β) · semantic + β · temporal
```

**β ∈ [0, 1].** It is an interpolation weight between two scores that are each in
[0, 1] — β = 0 is pure semantic search, β = 1 pure temporal. This is a different
parameter from the β of the older Gaussian kernel `exp(−β · Δφ²)`, which took values
in the thousands; that kernel remains available as
`temporal_encoding.temporal_alignment` but no longer drives ranking. The API rejects
anything outside the unit interval with HTTP 422
(`test_api.py::TestTemporalSearch::test_beta_is_bounded_to_the_unit_interval`).

β never admits a non-overlapping document: the gate runs before the blend, so no
value of β can recover a rejected candidate
(`test_retrieval.py::TestBeta`).

### 3.8 An unconstrained query is a full span, not "now"

A query with no temporal constraint is encoded as `full_span_interval()` — one whole
revolution of the outermost circle. Every arc saturates, the gate rejects nothing,
and ranking falls back to semantics. It is emphatically *not* encoded as a point at
the present moment, which would reject the entire historical corpus. "I did not say
when" and "I mean right now" are different questions
(`test_retrieval.py::TestUnconstrainedQuery`).

### 3.9 Natural-language query decomposition

One question can carry several independent temporal constraints. *"What was the Q1
impact on the full year for 2021, 2022 and 2023?"* is three anchor years × two
granularities = six retrievals, not one. `query_decomposition.decompose` produces the
sub-queries, `search_decomposed` runs them in parallel and merges on `group_id`,
recording `matched_subqueries` on each result. `MAX_SUBQUERIES` caps the fan-out and
sets `truncated`.

Disambiguation that matters: a bare "and" makes a **list** ("2022 and 2023" is two
years), while "between"/"from" makes a **range**. Intervals are half-open, so a range
label names the inclusive last year — `"2017-2022"` for an interval ending
2023-01-01, never `end.year`.

---

## 4. Conventions the code holds to

- **Half-open intervals `[start, end)`**, without exception. Q1 2026 is
  [1 Jan, 1 Apr) — 90 days, midpoint 15 Feb. An inverted or zero-length interval is
  rejected at the API boundary with HTTP 400.
- **Floored modulo**, remainder in `[0, modulus)`, exactly zero on a period
  boundary. The claim language "always positive" is precise only if zero counts as
  positive; the implementation's contract is *non-negative*, and
  `test_the_remainder_is_non_negative_and_below_the_modulus` states it that way.
- **Coverage is finite.** The shipped hierarchy covers epoch + 256 years, through
  2156. `TemporalHierarchy.covers()` answers for a given moment; beyond it, append
  `MILLENNIUM_SCALE` and re-index (§3.1, §3.5).
- **The calendar is proleptic Gregorian**, because that is what `datetime` is. Leap
  years follow the Gregorian rule everywhere, so 1900 and 2100 are common years and
  2000 is not. No carve-out is needed for the Julian calendar or the 1582/1752
  changeover: the default window starts at 1900, and `TemporalHierarchy.covers()`
  rejects anything outside it. A corpus that deliberately moves the epoch back past
  1582 will encode historical dates on the proleptic Gregorian axis, which is
  internally consistent but offset from the Julian dates in the source material by
  ten to thirteen days — an ingestion-side conversion, not an encoding one.
- **Weights are relative.** 0.4 / 0.5 / 0.1 are normalised over whichever scales a
  query actually traverses, so skipping a circle does not silently shrink the
  temporal score.

---

## 5. Where to look

| Concern | Module |
|---|---|
| epoch, scales, hierarchy, segments, floored modulo | `temporal_config.py` |
| intervals, phases, arcs, splitting, padding, traversal | `temporal_encoding.py` |
| document / query / result records | `temporal_spin.py` |
| embedding + encode + store | `ingestion.py` |
| memory, Chroma, pgvector backends | `vector_store.py` |
| two-pass search and scoring | `retrieval.py` |
| natural-language decomposition | `query_decomposition.py` |
| HTTP service | `api.py` |
