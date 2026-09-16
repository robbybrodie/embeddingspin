# Temporal-Phase Spin Retrieval

**Hierarchical phase-encoded temporal vectors for semantic retrieval.** Time is
mapped to a continuous geometric phase on several concurrent circular scales and
appended to a frozen semantic embedding, so a vector database can answer *when* as
precisely as it answers *what* — with no retraining and no metadata filter.

Covered by pending patent application **20251253US**. See
[PATENT_ALIGNMENT.md](PATENT_ALIGNMENT.md) for the claim-by-claim map to the code.

```bash
pip install -r requirements.txt
python demo.py          # end-to-end walkthrough, mock embeddings, no network
python arc_demo.py      # the geometry on its own
pytest                  # 385 tests
```

---

## The problem

"IBM Q2 2023 revenue" and "IBM Q2 2024 revenue" have nearly identical semantic
embeddings. A vector store ranks them the same and returns whichever happened to
land first. The usual answers all have costs:

| Approach | Cost |
|---|---|
| Metadata filters | Brittle. Exact-match only, no partial overlap, no ranking signal. |
| A timestamp dimension | Not periodic. Two adjacent quarters differ by a rounding error; a century differs by a lot. No natural notion of "inside". |
| Fine-tuning on time | Expensive, and time keeps moving. |

## The approach

Map a moment to an **angle**, and a duration to an **arc**, on several circles at
once:

```
φ = 2π · fmod_floored( (t − t₀) / T , 1.0 )
```

`t₀` is the base epoch, `T` the period of the circle. The modulo is applied to the
period *ratio*, before multiplication by 2π, so the operation stays dimensionless.
The remainder is floored — always in `[0, 1)`, exactly zero on a period boundary.

Each circle contributes three numbers:

```
[cos φ, sin φ, z]
```

`φ` is the phase of the **midpoint** of the interval; `z` is its arc length in
radians, capped at 2π. `z = 0` means a point in time. **N** circles give **3N**
dimensions, appended to the semantic embedding.

### N circles — three by default

The method is not three circles. The hierarchy is an ordered, variable-length tuple
of `ScaleSpec`, finest to coarsest, and every part of the system reads its length
from the stored header rather than assuming it. One circle works; so do five. What
ships is three, because three covers ordinary business time:

| Scale | Period | Divided into | Weight | What it discriminates |
|---|---|---|---|---|
| `quarter` | 1 year | 4 calendar quarters | 0.4 | position within the year |
| `decade` | 16 years | 16 calendar years | 0.5 | which year |
| `century` | 256 years | 16 sixteen-year blocks | 0.1 | which era |

Add or remove circles to suit the corpus:

```python
from temporal_config import (
    DEFAULT_HIERARCHY, TemporalHierarchy,
    QUARTER_SCALE, DECADE_SCALE, CENTURY_SCALE, MILLENNIUM_SCALE,
)

two   = TemporalHierarchy(scales=(QUARTER_SCALE, DECADE_SCALE))    # 6 dims, to 1916
four  = TemporalHierarchy(scales=(QUARTER_SCALE, DECADE_SCALE,
                                  CENTURY_SCALE, MILLENNIUM_SCALE))  # 12 dims, to 5996

# Or append to an existing one, which preserves every phase already stored:
four = DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE)
```

Coverage is always the epoch plus one revolution of the **outermost** circle, so the
scale set is how you buy range: 1916 with two circles, 2156 with three, 5996 with
four. Dimensionality is `3 × len(scales)`; segment counts and weights are per-scale.

Each period is an integer power of two and a factor of the next. That is not
decoration: it is what makes a boundary on a finer circle also a boundary on a
coarser one, which in turn is what lets interval splitting happen once, at the
finest scale, and stay valid everywhere above it — whatever the finest scale
happens to be.

### Worked example

```python
>>> from temporal_encoding import TemporalInterval, encode_single
>>> from temporal_config import DEFAULT_HIERARCHY
>>> interval = TemporalInterval.of_quarter(2026, 1)   # [1 Jan, 1 Apr), 90 days
>>> interval.midpoint.date()
datetime.date(2026, 2, 15)
>>> encoding = encode_single(interval, DEFAULT_HIERARCHY)
>>> quarter = encoding.tuple_for("quarter")
>>> round(quarter.cos, 4), round(quarter.sin, 4), round(quarter.z, 4)
(0.7147, 0.6995, 1.5493)
>>> len(encoding.to_vector())
9
```

45 days elapsed of a 365-day year puts the midpoint at 0.7746 rad; the 90-day span
is 1.5493 rad of arc. The full nine-dimensional vector under the shipped epoch:

```
quarter   cos  0.7147  sin  0.6995  z 1.5493     segment Q1
decade    cos  0.7405  sin -0.6721  z 0.0968     segment: year +14y
century   cos -0.9989  sin  0.0460  z 0.0061     segment: 16-year block +112y
```

---

## What the system does

### Epoch

`t₀` defaults to **1900-01-01 UTC**, which covers through **2156** — the epoch plus
one revolution of the 256-year circle. Override with `EMBEDDINGSPIN_EPOCH`; it is
read once, at import.

**Changing the epoch is a re-index.** A stored phase survives an epoch shift only if
the shift is a whole multiple of *every* period. 2010 → 1900 is 110 years: a
multiple of 1, but not of 16 or 256.

```python
from temporal_config import (
    DEFAULT_HIERARCHY, PATENT_EXAMPLE_HIERARCHY, describe_epoch_migration,
)

report = describe_epoch_migration(PATENT_EXAMPLE_HIERARCHY, DEFAULT_HIERARCHY)
report["requires_reindex"]   # True
report["reasons"]            # what changed, and why it matters
```

`epoch_shift_is_congruent(old_epoch, new_epoch, scales)` answers the narrower
question directly.

`TemporalHierarchy.fingerprint()` identifies the configuration a corpus was written
under, and the store refuses to serve a corpus whose fingerprint does not match the
retriever's:

```
v2|1900-01-01|calendar|quarter:1:4+decade:16:16+century:256:16
```

The legacy 2010 epoch is kept as `PATENT_EXAMPLE_HIERARCHY` so the numbers published
in the application stay reproducible.

### Calendar years, not average years

Position within the 1-year circle is a fraction of the **actual** calendar year — 365
or 366 days — so 1 January is phase 0 in every year, leap years included. The older
fixed-length convention (365.2425 days) is retained as `linear` via
`EMBEDDINGSPIN_YEAR_CONVENTION` for corpora already written under it.

### Segments — even geometry, calendar identity

Every circle is divided **evenly**: four 90° arcs on the 1-year circle, sixteen 22.5°
arcs on the others. Periods and segment counts are both powers of two, so a segment
index is

```
index = int(fraction * segments)        # fraction is already in [0, 1)
```

— one multiply and a truncation. No divider table, no search, no data-dependent
branch, and the same arithmetic vectorises across a whole batch of candidates. That
is the reason the periods are 1 / 16 / 256 rather than 1 / 10 / 100, and it is the
reason the segment counts are 4 / 16 / 16. The geometry is not allowed to become
uneven, because evenness is what the scale performance rests on.

#### The calendar is uneven, and it is handled afterwards

Quarters run 90, 91, 92 and 92 days, and 29 February sits inside Q1, so in a leap
year every later divider shifts one ordinal day out. Measured against its own year's
length, 1 April is `91/366 = 0.24863` in a leap year against `90/365 = 0.24658` in a
common one — 18 hours apart at the Q2 divider, 12 at Q3, 6 at Q4.

None of that enters the geometry. Segment **identity** is resolved from the real
dates, once, at encoding time — the last point at which the dates are still in hand —
by `temporal_encoding.calendar_segment` and `calendar_segments_touched`:

```
quarter circle  : (month - 1) // 3
16-year circle  : (year - epoch_year) % 16
256-year circle : ((year - epoch_year) // 16) % 16
```

Integer arithmetic, exact by construction. There is no divider to approximate, so the
uneven quarters and the leap day need no special case, no second table and no
tolerance. The answer is stored on the tuple and never re-derived downstream.

Comparison sees only phases, so it uses the even windows. `segment_of_phase` and
`calendar_segment` therefore disagree near a divider by design — 1 April is
geometrically Q1 and calendrically Q2 — and that disagreement is left alone. It
cannot admit a false positive, because the overlap gate is the arc intersection, not
the segment; an even window can only adjust a per-segment score, and measured across
every quarter and month pair of 2023 and 2024 the adjustment is nil.

`tests/test_segments.py` sweeps **every hour** of eight sample years spanning
1900–2155, and every month and quarter of every year from 1900 to 2156, on all three
circles.

Segments also carry scoring. Each shared segment is scored with its own Jaccard
coefficient, restricted to that segment's window:

```
jaccard = max( whole_arc_jaccard , max over shared segments )
```

A query window straddling a divider — running from inside 2022 into 2023 — therefore
still reaches relevant documents on the far side, rather than losing them because the
query's centre of mass sat on the near side.

### Boundary splitting

An interval crossing a divider on the finest circle is stored as several
representations, one per period it spans, all sharing a `group_id`:

```
ibm-report-2017-2022#0    2017-01-01 → 2018-01-01    decade φ = 112.5°
ibm-report-2017-2022#1    2018-01-01 → 2019-01-01    decade φ = 135.0°
ibm-report-2017-2022#2    2019-01-01 → 2020-01-01    decade φ = 157.5°
ibm-report-2017-2022#3    2020-01-01 → 2021-01-01    decade φ = 180.0°
ibm-report-2017-2022#4    2021-01-01 → 2022-01-01    decade φ = 202.5°
ibm-report-2017-2022#5    2022-01-01 → 2023-01-01    decade φ = 225.0°
```

Each component has a full 1-year arc and its own year on the 16-year circle, so a
query about 2021 reaches the review through exactly one of them. Retrieval collapses
them on `group_id`, so the consumer sees the document once. `MAX_REPRESENTATIONS`
caps the fan-out; past it, the interval is kept whole.

This is why `/stats` reports `total_representations` and `total_documents`
separately, and why `/ingest` returns both `ingested_count` and
`representation_count`.

### Self-describing vectors

Every encoding carries a header: schema version, epoch, year convention,
`tuple_count`, and per tuple the period, segment count and weight. A reader strips
exactly `3 × tuple_count` trailing dimensions rather than assuming nine, so a store
can hold three-circle and four-circle vectors side by side and split each correctly.

Vectors written before an outer circle was appended are brought forward with a
**neutral tuple** `[cos = 1, sin = 0, z = 2π]` — a saturated arc that overlaps
everything and therefore cannot reject anything (`pad_to_hierarchy`).

Appending an outer scale preserves every existing phase, because a finer circle's
phase does not depend on the periods above it. Changing the epoch does not. Both are
true at once and the difference matters; see the epoch section above.

### Lazy resolution

A query descends only as far as it needs. A scale is **skipped** when the query arc
saturates that circle: a full-circle arc overlaps everything, so the check cannot
reject anything. "All of 2021" spans the entire 1-year circle, so the 1-year check is
dropped and the year is resolved on the 16-year circle instead.

The plan is ordered **coarsest first**, so the widest and cheapest rejections happen
before any fine-grained work.

| Query | Traversed |
|---|---|
| a single day | `century`, `decade`, `quarter` |
| a month or a quarter | `century`, `decade`, `quarter` |
| a full year | `century`, `decade` |
| 2017–2022 | `century`, `decade` |
| 2000–2050 | `century` |
| unconstrained | `century` |

Encoding precision and traversal depth are separate concerns. Precision is fixed at
ingestion — every scale is always written, nothing is discarded. Depth is a property
of the question. Retrieval cost tracks the precision the query asked for, not the
precision the corpus holds.

### Retrieval

**Pass 1** — coarse recall of 200 candidates over the concatenated vector, with the
temporal block scaled by λ = 0.1 so it nudges rather than dominates.

**Pass 2** — for each candidate, at each traversed scale, coarsest first:

1. **Hard gate.** No overlap at any traversed scale ⇒ the document is rejected, not
   down-weighted. This is what stops 2023 documents bleeding into a 2024 query.
2. **Soft score.** Per-scale Jaccard, weighted and normalised over the scales
   actually traversed.

```
temporal = Σ  w_s · jaccard_s
score    = (1 − β) · semantic + β · temporal
```

**Pass 3** — deduplicate on `group_id`, rank.

#### β ∈ [0, 1]

β is an interpolation weight between two scores that are each in [0, 1].

| β | Behaviour |
|---|---|
| 0.0 | pure semantic ranking (the gate still applies) |
| 0.5 | balanced — the default |
| 1.0 | pure temporal ranking |

This is **not** the β of the older Gaussian kernel `exp(−β · Δφ²)`, which took values
in the thousands. If you are carrying a β of 5000 forward from a previous version,
see [CHANGELOG.md](CHANGELOG.md). The API rejects anything outside the unit interval
with HTTP 422.

No value of β can recover a document the gate rejected — the gate runs first.

#### An unconstrained query is a full span, not "now"

A query with no temporal constraint is encoded as one whole revolution of the
outermost circle. Every arc saturates, the gate rejects nothing, and ranking falls
back to semantics. It is *not* encoded as a point at the present moment, which would
reject the entire historical corpus. "I did not say when" and "I mean right now" are
different questions.

### Natural-language decomposition

One question can carry several independent temporal constraints:

```python
from query_decomposition import decompose, search_decomposed

decompose("What was the Q1 impact on the full year for 2021, 2022 and 2023?")
# → Q1 2021, FY2021, Q1 2022, FY2022, Q1 2023, FY2023
```

Three anchor years × two granularities = six retrievals, not one. They run in
parallel and merge on `group_id`, each result recording the `matched_subqueries` it
answered.

A bare "and" makes a **list** — "2022 and 2023" is two years. "between" or "from"
makes a **range**. Intervals are half-open, so a range label names the inclusive last
year: `"2017-2022"` for an interval ending 2023-01-01.

---

## Data structures

```
TemporalInterval     half-open [start, end); end=None is an instant
  .midpoint          what cos/sin encode
  .of_year(2021)  .of_quarter(2021, 2)  .of_month(2021, 3)  .spanning(2017, 2022)

ScaleTuple           one circle: scale_name, cos, sin, z, phi_start, segments
TemporalEncoding     the tuples + hierarchy + group_id + representation index
  .to_vector()       flat [cos, sin, z] × tuple_count
  .header()          what a reader needs to split the vector back apart
  .to_dict() / .from_dict()

ScaleSpec            name, period_years, segments, weight, segment_label
                     (dividers are derived: i * 2π / segments)
TemporalHierarchy    the ordered scales + epoch + year convention
  .fingerprint()  .covers(moment)  .extended(scale)  .normalized_weights(names)

SpinDocument         doc_id, text, semantic_embedding, encoding,
                     full_embedding = semantic ++ temporal, group_id, metadata
SpinQuery            query_text, semantic_embedding, encoding, lambda_factor, plan
TraversalPlan        scale_names (coarsest first) + skipped, with reasons
ScaleMatch           scale_name, overlaps, jaccard, delta_phi,
                     shared_segments, segment_jaccard
RetrievalResult      doc_id, group_id, text, interval, semantic_score,
                     temporal_alignment, combined_score, scale_matches,
                     traversed_scales, rejected_at, rank, metadata
```

---

## Usage

### Ingest

```python
from datetime import datetime, timezone

from ingestion import create_ingestion_pipeline
from temporal_encoding import TemporalInterval
from vector_store import InMemoryVectorStore

pipeline = create_ingestion_pipeline(InMemoryVectorStore(), use_mock_embeddings=True)

# A period — stored as an arc
pipeline.ingest_document(
    "IBM fiscal year 2023: record AI growth.",
    interval=TemporalInterval.of_year(2023),
    doc_id="ibm-fy2023",
)

# A quarter inside it
pipeline.ingest_document(
    "Q2 2023 performance exceeded expectations.",
    interval=TemporalInterval.of_quarter(2023, 2),
    doc_id="ibm-q2-2023",
)

# An instant — z = 0 at every scale
pipeline.ingest_document(
    "IBM announced an acquisition on 15 March 2023.",
    interval=TemporalInterval.point(datetime(2023, 3, 15, tzinfo=timezone.utc)),
    doc_id="ibm-news-20230315",
)

# A span — split into six rows sharing one group_id
pipeline.ingest_document(
    "IBM six-year strategic review.",
    interval=TemporalInterval.spanning(2017, 2022),
    doc_id="ibm-review-2017-2022",
)
```

Omit `interval` and the pipeline extracts one from the text
(`temporal_spin.extract_interval_from_text` recognises phrases like *"for the period
ended 31 December 2019"*), falling back to file metadata and then ingestion time.

### Search

```python
from retrieval import TemporalSpinRetriever

retriever = TemporalSpinRetriever(client, store, default_beta=0.5)

results = retriever.search(
    "revenue growth",
    interval=TemporalInterval.of_quarter(2023, 2),
    beta=0.5,
)
# Reaches: the Q2 report, and the FY2023 report that contains it.
# Rejects:  Q2 2024 — no overlap on the 16-year circle.

for r in results:
    print(r.rank, r.doc_id, r.combined_score, r.traversed_scales)
    print(retriever.explain_result(r))
```

Unconstrained:

```python
retriever.search("IBM cloud strategy")        # full-span arc, semantic ranking
```

Several constraints at once:

```python
from query_decomposition import search_decomposed
results, decomposition = search_decomposed(
    retriever, "Q1 impact on the full year for 2021, 2022 and 2023", beta=0.5
)
```

### CLI

```bash
python demo.py                                        # full walkthrough
python demo.py --query "IBM cloud strategy"           # unconstrained
python demo.py --query "revenue" --start 2021-01-01 --end 2021-04-01 --beta 0.7
python demo.py --beta-sweep
python demo.py --decompose "Q1 impact on the full year for 2021, 2022 and 2023"
python arc_demo.py                                    # geometry, no retrieval
```

### HTTP

```bash
python api.py
open http://localhost:8080/docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /stats` | counts, plus the epoch, convention, scales and fingerprint the corpus was written under |
| `POST /ingest` | index chunks; returns `ingested_count` and `representation_count` |
| `POST /temporal_search` | two-pass search; returns `traversed_scales`, `skipped_scales` and a per-scale breakdown |
| `POST /decompose` | decomposition only, no retrieval |
| `POST /decomposed_search` | decompose, retrieve in parallel, merge |
| `POST /beta_sweep` | one query at several β values, without re-indexing |
| `POST /clear` | empty the store |

```bash
curl -X POST localhost:8080/temporal_search -H 'Content-Type: application/json' -d '{
  "query": "IBM revenue",
  "start": "2021-01-01",
  "end":   "2021-04-01",
  "beta":  0.5,
  "top_k": 10
}'
```

Intervals are half-open. `end` is exclusive; omit it for an instant. An inverted or
zero-length interval is a 400.

---

## Configuration

### Temporal encoding

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDINGSPIN_EPOCH` | `1900-01-01` | Read once at import. Changing it on an existing corpus requires a re-index. |
| `EMBEDDINGSPIN_YEAR_CONVENTION` | `calendar` | `calendar` (actual 365/366-day years) or `linear` (fixed 365.2425). |

### Service

| Variable | Default | Notes |
|---|---|---|
| `USE_OPENAI_EMBEDDINGS` | `true` | Needs `OPENAI_API_KEY`. |
| `USE_MOCK_EMBEDDINGS` | `false` | Deterministic, offline. |
| `LLAMASTACK_URL` | `http://localhost:8000` | Used when neither of the above is set. |
| `LLAMASTACK_API_KEY` | – | Optional. |
| `EMBEDDING_MODEL` | `text-embedding-3-large` / `text-embedding-v1` | Per backend. |
| `VECTOR_STORE` | `memory` | `memory`, `chroma`, `pgvector`. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | |
| `DATABASE_URL` | – | pgvector connection string. |
| `DEFAULT_BETA` | `0.5` | Must be in [0, 1]. |
| `LOAD_DEMO_DATA` | `true` | Loads the IBM corpus on startup if the store is empty. |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | |

Production against LlamaStack and pgvector:

```bash
export USE_OPENAI_EMBEDDINGS=false USE_MOCK_EMBEDDINGS=false
export LLAMASTACK_URL=$MODEL_GATEWAY_URL
export VECTOR_STORE=pgvector DATABASE_URL=$POSTGRES_CONNECTION_STRING
python api.py
```

See [PREREQUISITES.md](PREREQUISITES.md) for installing the backends.

---

## Testing

```bash
pytest                                   # 385 tests
pytest tests/test_patent_conformance.py  # the published behaviour specifically
pytest tests/test_segments.py            # even geometry vs calendar identity
```

| Module | Covers |
|---|---|
| `test_temporal_config.py` | floored modulo, scale validation, hierarchy header, epoch migration |
| `test_temporal_encoding.py` | intervals, phases, arc algebra, splitting, padding, traversal |
| `test_segments.py` | even divider geometry, calendar identity from the dates, every hour of eight sample years, every month and quarter 1900–2156 |
| `test_patent_conformance.py` | Formula 1, the published worked example, vector structure, lazy resolution, extensibility, hierarchies of 1–5 circles |
| `test_vector_store.py` | round-trip, header-driven splitting, groups, filters, hierarchy guard |
| `test_retrieval.py` | the gate, multi-year documents, dedup, β, unconstrained queries |
| `test_query_decomposition.py` | anchors, granularities, lists vs ranges, merged search |
| `test_api.py` | the service end to end |

`conftest.py` pins the epoch and the year convention before `temporal_config` is
imported, because the epoch is read at import time.

---

## Project structure

```
temporal_config.py       epoch, scales, hierarchy, segments, floored modulo
temporal_encoding.py     intervals, phases, arcs, splitting, padding, traversal
temporal_spin.py         SpinDocument / SpinQuery / RetrievalResult, text extraction
ingestion.py             embed + encode + store
llamastack_client.py     LlamaStack Model Gateway client, plus MockEmbeddingClient
openai_client.py         OpenAI embeddings, same interface
vector_store.py          in-memory, Chroma, pgvector
retrieval.py             two-pass search and scoring
query_decomposition.py   natural-language decomposition
api.py                   FastAPI service
demo.py / arc_demo.py    walkthroughs
demo_data.py             mock IBM corpus, 2015–2024
tests/                   385 tests
PATENT_ALIGNMENT.md      claims → implementation → tests
CHANGELOG.md             v1 → v2, and how to migrate
```

---

## Design notes

**Why circles.** Time that repeats — quarters, years, blocks of years — is naturally
angular. A circle gives bounded coordinates, smooth interpolation, wraparound for
free, and a geometric meaning for "inside", "overlaps" and "how far apart".

**Why several circles.** One circle cannot separate *which quarter* from *which
year*. Two moments a year apart are adjacent on a 256-year circle and coincident on a
1-year one; only the 16-year circle sees them as distinct. Concurrent scales give the
system a different question to ask at each resolution.

**Why powers of two, each a factor of the next.** Two reasons. A boundary on a finer
circle is then always a boundary on a coarser one, so splitting can happen once, at
the finest scale, and remain correct above it. And the segment index becomes
`int(fraction * segments)` — a multiply and a truncation over a power of two, with no
table, no search and no data-dependent branch, so it vectorises across a batch. That
second reason is why the calendar's unevenness is kept out of the geometry entirely
and applied afterwards, at ingestion.

**Why arcs and not just points.** A quarterly filing is not an instant, and a
question about a quarter is not a question about a day. Encoding extent is what lets
a 10-Q answer a question phrased about the 10-K that contains it.

**Why the model stays frozen.** The temporal block is appended after embedding. Any
embedding model works, nothing is retrained, and the temporal focus stays adjustable
at query time.

**What this does not do.** Coverage is finite — epoch plus 256 years, through 2156
under the shipped configuration. Moving the epoch or changing the scale set is a
re-index, and the code says so rather than implying otherwise.

---

## credits

Robby Brodie — robbytherobot@redhat.com

- **Bryon Baker** (brbaker@redhat.com) and **Joe Wood** for collaborative development of the
  arc-based temporal encoding, which solved hierarchical time-period matching for
  financial reporting (10-Q ⊂ 10-K) and time-series chunking.
- Red Hat AI 3 (LlamaStack) for the Model Gateway API.
- pgvector and Chroma for vector search; dateutil and FastAPI for the rest.
- A special mention to Matt Hicks for his problem statement definition and encouragement
