# Changelog

## 2.0.0 — alignment with patent application 20251253US

A breaking rewrite of the temporal layer. Vectors written by 1.x cannot be read by
2.0 and must be re-indexed.

The three development notes that used to live here — `ARC_ENCODING_SUMMARY.md`,
`FINAL_SUMMARY.md` and `ZOOM_FIX_SUMMARY.md` — described the 1.x design and
contradicted this one on nearly every number. They were removed; their content is
in git history, and what survives of it is below.

### Breaking

**β now means something else.** In 1.x, β was the width parameter of a Gaussian
kernel `exp(−β · Δφ²)` and typical values were in the thousands (the default was
5000). In 2.0, β is an interpolation weight in **[0, 1]**:

```
score = (1 − β) · semantic + β · temporal
```

0 is pure semantic, 1 is pure temporal, 0.5 is the default. The API rejects anything
outside the unit interval with HTTP 422. **If you are carrying a β of 5000 forward,
it is not a smaller number now — it is a different parameter.** Start at 0.5.

The Gaussian kernel still exists as `temporal_encoding.temporal_alignment` for
callers that want it, but it no longer drives ranking; per-scale Jaccard does.

**The epoch moved from 2010 to 1900.** That shift is 110 years — a multiple of the
1-year period but not of the 16-year or 256-year periods — so every coarse phase in
a 1.x corpus moves. Re-index. The old epoch is preserved as
`PATENT_EXAMPLE_HIERARCHY` (coverage through 2266) so the worked example published in
the application stays reproducible.

**Intervals are half-open `[start, end)`, without exception.** 1.x was inconsistent
about whether an end date was included. Q1 2026 is now [1 Jan, 1 Apr) — 90 days,
midpoint 15 February. An inverted or zero-length interval is a 400 at the API
boundary.

**`compute_spin_vector` is superseded** by `temporal_encoding.encode` /
`encode_single`, which return a `TemporalEncoding` rather than a bare list. The old
function remains as a thin shim.

**Ingestion and search take an `interval`, not a `timestamp` / `end_timestamp`
pair.** The old field names are still accepted at the HTTP boundary as deprecated
aliases.

**Storage is header-driven.** A reader strips `3 × tuple_count` trailing dimensions
from the stored vector, using the header written alongside it, rather than assuming
nine. 1.x rows have no header and cannot be read.

### Added

- **Floored modulo** (`floored_mod`), applied to the period ratio before
  multiplication by 2π. The remainder is in `[0, 1)` and is exactly zero on a period
  boundary.
- **Calendar year convention.** Position within the 1-year circle is a fraction of
  the actual 365- or 366-day year, so 1 January is phase 0 in every year. The old
  fixed 365.2425-day behaviour is available as `EMBEDDINGSPIN_YEAR_CONVENTION=linear`.
- **Configurable epoch** via `EMBEDDINGSPIN_EPOCH`, with
  `epoch_shift_is_congruent()` and `describe_epoch_migration()` to tell you in
  advance whether a change costs a re-index.
- **Hierarchy fingerprints** and a compatibility guard: a store and a retriever
  configured differently now fail loudly instead of returning quiet nonsense.
- **Segments — even geometry.** Four quarters on the 1-year circle, sixteen years on
  the 16-year circle, sixteen 16-year blocks on the 256-year circle. Every circle is
  divided evenly and the segment counts are powers of two alongside the periods, so a
  segment index is `int(fraction * segments)` — a multiply and a truncation, with no
  divider table, no search and no data-dependent branch, and it vectorises across a
  batch. `ScaleSpec` therefore carries no boundaries and no tolerance: dividers are
  derived as `i * 2π / segments`.
- **Calendar segment identity, resolved at ingestion.** `calendar_segment` and
  `calendar_segments_touched` answer *which real quarter, which real year* from the
  dates themselves, in integer arithmetic — `(month - 1) // 3`,
  `(year - epoch_year) % 16`, `((year - epoch_year) // 16) % 16` — at encoding time,
  the last point at which the dates are in hand. The answer is stored on the tuple
  and never re-derived. The calendar is uneven (quarters of 90/91/92/92 days; 29
  February shifts every later divider one ordinal day out, which is a *different
  phase* because each year is measured against its own length), and none of that is
  allowed into the geometry. Comparison, which sees only phases, uses the even
  windows; the two disagree near a divider by design, which cannot admit a false
  positive because the gate is the arc intersection, not the segment.
- **Per-segment Jaccard.** Each shared segment is scored in its own window and a
  match on any of them qualifies the document, so a query straddling a divider no
  longer loses everything on the far side.
- **Zero-degree boundary splitting.** A multi-year interval is stored once per
  period it spans, all rows sharing a `group_id`, deduplicated at retrieval.
  `MAX_REPRESENTATIONS` caps the fan-out.
- **N circles, not three.** The hierarchy is an ordered variable-length tuple of
  `ScaleSpec`; 1 / 16 / 256 years is the default, not the method. The vector width
  (`3 × len(scales)`), the split point (`hierarchy.finest`), the traversal order, the
  weight normalisation and the coverage horizon all derive from the scale set.
  Hierarchies of one through five circles are exercised in
  `test_patent_conformance.py::TestArbitraryScaleCount`, including a four-circle
  corpus searched end to end.
- **Variable-length self-describing vectors** and **neutral padding**
  (`[cos = 1, sin = 0, z = 2π]`) so an outer circle can be appended without
  invalidating existing rows. Prefix compatibility means a three-circle retriever can
  read a four-circle corpus without re-encoding.
- **`MILLENNIUM_SCALE`** (4096 years), an optional fourth circle for corpora that
  need to reach past 2156.
- **Lazy resolution.** `traversal_plan()` skips any circle the query arc saturates
  and orders the rest coarsest-first, so the cheapest rejections happen first.
  Encoding precision (fixed at ingestion) and traversal depth (a property of the
  question) are deliberately separate.
- **Natural-language query decomposition** (`query_decomposition.py`): one question
  with several temporal constraints becomes several retrievals, run in parallel and
  merged on `group_id` with `matched_subqueries` recorded. Exposed as `/decompose`
  and `/decomposed_search`.
- **A test suite.** 385 tests across eight modules, including
  `test_patent_conformance.py`, which reproduces Formula 1 longhand and the published
  worked example to 5e-5.
- `PATENT_ALIGNMENT.md`, mapping every claim to its implementation and its test.

### Fixed

- **An unconstrained query returned nothing.** `create_query` documented a full-span
  arc but encoded a point at `datetime.now()`, which the overlap gate then used to
  reject the entire historical corpus. It now encodes `full_span_interval()`. "I did
  not say when" and "I mean right now" are different questions.
- **Leap-year phase drift.** Under the old fixed-length year, 1 January was not
  phase 0 — the error accumulated across leap years. It is now exactly 0 for every
  year from 1900 to 2155.
- **Quarters were mislabelled near a divider, and in a leap year by up to 36 hours.**
  Reading a calendar quarter off a phase gets 1 April wrong either way: against even
  90° arcs it is at 88.8° and reads as Q1, and against a fixed table of calendar
  dividers the leap-year shift makes the last 18 hours of 31 March, 12 of 30 June and
  6 of 30 September read one quarter too late. Whole dates and all month, quarter and
  year intervals were correct, and the overlap gate — which is geometric — was
  correct throughout, so retrieval was unaffected; the wrong value reached callers
  through the `segment_label` field, which `api.py` surfaces. Fixed by not deriving
  identity from a phase at all: `calendar_segment` works from the dates in integer
  arithmetic, which is exact for every hour of every year in the coverage window.
- **A moment exactly on a divider could fall on the wrong side** in a handful of
  years (2048, 2052, …), because the divider phase was reached two ways — an ordinal
  day over the year length, and a difference of two absolute year positions — which
  disagree in the last few bits. There is no float comparison at a calendar divider
  any more, so the failure mode is gone rather than tolerated.
- **Deprecated FastAPI `@app.on_event` handlers** replaced with a lifespan context
  manager.
- Several unused imports removed from `retrieval.py`.

### Migration from 1.x

1. Re-index. The epoch, the year convention, the vector layout and the storage
   header all changed; nothing carries over.
2. Replace `timestamp` / `end_timestamp` with a `TemporalInterval`. Remember the end
   is exclusive: a full year is `TemporalInterval.of_year(2023)`, not
   1 Jan → 31 Dec.
3. Set β to 0.5 and adjust within [0, 1]. Whatever value you used before does not
   translate.
4. Expect `/stats` and `/ingest` to report representations and documents separately
   once you index anything spanning more than one year.
5. If you need a 2010 epoch, construct `PATENT_EXAMPLE_HIERARCHY` explicitly rather
   than relying on the default.

Moving segment identity out of the geometry does **not** change the fingerprint,
which is `name:period:segments` and so is unaffected by the removal of the divider
tables. No `[cos, sin, z]` value changes either — only the segment list stored
alongside them. A corpus written earlier keeps the old identity for the ~36 hours per
leap year that were affected, which costs those rows the per-segment scoring bonus
and mislabels them in API responses. Nothing is rejected and nothing is mis-ranked,
so re-encoding is optional; if you want it, re-ingest and the store will accept the
rows in place.

---

## 1.x — historical

Single circle with a configurable period (1000 years by default), a 2D
`[cos φ, sin φ]` spin vector, and re-ranking by `semantic × exp(−β · Δφ²)` with β in
the thousands. Later 1.x added a third component per scale for arc length and moved
to three concurrent circles at 1 / 16 / 256 years, which is the design 2.0 builds on.
