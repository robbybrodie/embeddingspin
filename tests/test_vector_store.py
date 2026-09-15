"""
Storage round-trip.

The storage layer is where the variable-length, self-describing vector has to
survive being flattened into a backend that only knows about floats and strings.
The critical property is that the temporal block is recovered from the stored
header — never by assuming nine trailing dimensions — so a corpus written under a
three-circle hierarchy and a corpus written under a four-circle one can be read by
the same code.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from temporal_config import DEFAULT_HIERARCHY, MILLENNIUM_SCALE, TemporalHierarchy
from temporal_encoding import TemporalEncoding, TemporalInterval, encode, encode_single
from temporal_spin import SpinDocument
from vector_store import InMemoryVectorStore


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def make_doc(doc_id, interval, semantic=None, group_id=None, metadata=None, hierarchy=None):
    hierarchy = hierarchy or DEFAULT_HIERARCHY
    encoding = encode_single(interval, hierarchy, group_id=group_id or doc_id)
    return SpinDocument(
        doc_id=doc_id,
        text=f"text for {doc_id}",
        semantic_embedding=semantic or [0.1, 0.2, 0.3, 0.4],
        encoding=encoding,
        metadata=metadata or {},
    )


@pytest.fixture
def memory_store():
    return InMemoryVectorStore(hierarchy=DEFAULT_HIERARCHY)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_the_full_embedding_is_semantic_then_temporal(self):
        doc = make_doc("a", TemporalInterval.of_quarter(2026, 1))
        assert doc.full_embedding == doc.semantic_embedding + doc.encoding.to_vector()
        assert len(doc.full_embedding) == 4 + 9

    def test_a_document_survives_storage_and_retrieval(self, memory_store):
        doc = make_doc("a", TemporalInterval.of_quarter(2026, 1))
        memory_store.add_documents([doc])
        restored = memory_store.get_document("a")
        assert restored is not None
        assert restored.doc_id == "a"
        assert restored.text == doc.text
        assert restored.group_id == doc.group_id

    def test_the_temporal_vector_survives_exactly(self, memory_store):
        doc = make_doc("a", TemporalInterval.of_quarter(2026, 1))
        memory_store.add_documents([doc])
        restored = memory_store.get_document("a")
        assert restored.encoding.to_vector() == pytest.approx(doc.encoding.to_vector())

    def test_the_interval_survives(self, memory_store):
        interval = TemporalInterval(utc(2021, 3, 4), utc(2021, 9, 8))
        memory_store.add_documents([make_doc("a", interval)])
        restored = memory_store.get_document("a")
        assert restored.interval.start == interval.start
        assert restored.interval.end == interval.end

    def test_a_point_round_trips_as_a_point(self, memory_store):
        memory_store.add_documents(
            [make_doc("p", TemporalInterval.point(utc(2023, 5, 10)))]
        )
        restored = memory_store.get_document("p")
        assert restored.interval.is_point
        assert not restored.is_arc

    def test_metadata_survives(self, memory_store):
        memory_store.add_documents(
            [make_doc("a", TemporalInterval.of_year(2021), metadata={"type": "10-K", "year": 2021})]
        )
        restored = memory_store.get_document("a")
        assert restored.metadata["type"] == "10-K"
        assert restored.metadata["year"] == 2021

    def test_reserved_metadata_keys_cannot_be_overwritten(self, memory_store):
        """A caller passing ``group_id`` in metadata must not clobber the real one."""
        doc = make_doc(
            "a",
            TemporalInterval.of_year(2021),
            group_id="real-group",
            metadata={"group_id": "spoofed", "temporal_encoding": "garbage"},
        )
        memory_store.add_documents([doc])
        restored = memory_store.get_document("a")
        assert restored.group_id == "real-group"
        assert restored.encoding.to_vector() == pytest.approx(doc.encoding.to_vector())


# ---------------------------------------------------------------------------
# Header-driven split
# ---------------------------------------------------------------------------


class TestHeaderDrivenSplit:
    def test_the_semantic_half_is_recovered_using_the_stored_tuple_count(self, memory_store):
        semantic = [0.5] * 7
        memory_store.add_documents(
            [make_doc("a", TemporalInterval.of_year(2021), semantic=semantic)]
        )
        restored = memory_store.get_document("a")
        assert restored.semantic_embedding == pytest.approx(semantic)

    def test_a_four_circle_vector_is_split_at_twelve_not_nine(self):
        """
        The number of trailing dimensions is read from the header. Assuming nine
        would leave three temporal components stranded in the semantic half.
        """
        extended = TemporalHierarchy().extended(MILLENNIUM_SCALE)
        store = InMemoryVectorStore(hierarchy=extended)
        semantic = [0.25] * 5
        store.add_documents(
            [make_doc("a", TemporalInterval.of_year(2021), semantic=semantic, hierarchy=extended)]
        )
        restored = store.get_document("a")
        assert len(restored.encoding.tuples) == 4
        assert restored.semantic_embedding == pytest.approx(semantic)
        assert len(restored.full_embedding) == 5 + 12

    def test_the_stored_header_describes_the_hierarchy(self, memory_store):
        memory_store.add_documents([make_doc("a", TemporalInterval.of_year(2021))])
        restored = memory_store.get_document("a")
        assert restored.encoding.hierarchy.fingerprint() == DEFAULT_HIERARCHY.fingerprint()


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


class TestGroups:
    def test_split_representations_are_stored_separately_but_grouped(self, memory_store):
        encodings = encode(TemporalInterval.spanning(2017, 2022), group_id="review")
        docs = [
            SpinDocument(
                doc_id=f"review#{i}",
                text="strategic review",
                semantic_embedding=[0.1, 0.2],
                encoding=enc,
            )
            for i, enc in enumerate(encodings)
        ]
        memory_store.add_documents(docs)

        assert memory_store.count() == 6
        assert memory_store.count_groups() == 1
        assert len(memory_store.get_group("review")) == 6

    def test_count_and_count_groups_differ_when_documents_split(self, memory_store):
        for year in (2020, 2021):
            memory_store.add_documents([make_doc(f"y{year}", TemporalInterval.of_year(year))])
        encodings = encode(TemporalInterval.spanning(2017, 2018), group_id="span")
        memory_store.add_documents([
            SpinDocument(f"span#{i}", "span", [0.1, 0.2], enc)
            for i, enc in enumerate(encodings)
        ])
        assert memory_store.count() == 4
        assert memory_store.count_groups() == 3

    def test_an_unknown_group_is_empty(self, memory_store):
        assert memory_store.get_group("nope") == []


# ---------------------------------------------------------------------------
# Search and filtering
# ---------------------------------------------------------------------------


class TestSearchAndFilter:
    @pytest.fixture
    def populated(self, memory_store):
        memory_store.add_documents([
            make_doc("a", TemporalInterval.of_year(2021), semantic=[1.0, 0.0],
                     metadata={"type": "10-K", "year": 2021}),
            make_doc("b", TemporalInterval.of_year(2022), semantic=[0.0, 1.0],
                     metadata={"type": "10-Q", "year": 2022}),
            make_doc("c", TemporalInterval.of_year(2023), semantic=[0.7, 0.7],
                     metadata={"type": "10-K", "year": 2023}),
        ])
        return memory_store

    def test_search_returns_the_nearest_first(self, populated):
        query = [1.0, 0.0] + encode_single(TemporalInterval.of_year(2021)).to_vector()
        results = populated.search(query, top_k=3)
        assert results[0][0].doc_id == "a"

    def test_top_k_is_respected(self, populated):
        query = [1.0, 0.0] + encode_single(TemporalInterval.of_year(2021)).to_vector()
        assert len(populated.search(query, top_k=2)) == 2

    def test_equality_filter(self, populated):
        query = [1.0, 0.0] + encode_single(TemporalInterval.of_year(2021)).to_vector()
        results = populated.search(query, top_k=10, filter_dict={"type": "10-K"})
        assert {doc.doc_id for doc, _ in results} == {"a", "c"}

    def test_in_filter(self, populated):
        query = [1.0, 0.0] + encode_single(TemporalInterval.of_year(2021)).to_vector()
        results = populated.search(
            query, top_k=10, filter_dict={"year": {"$in": [2021, 2023]}}
        )
        assert {doc.doc_id for doc, _ in results} == {"a", "c"}

    def test_ne_filter(self, populated):
        query = [1.0, 0.0] + encode_single(TemporalInterval.of_year(2021)).to_vector()
        results = populated.search(query, top_k=10, filter_dict={"type": {"$ne": "10-K"}})
        assert {doc.doc_id for doc, _ in results} == {"b"}

    def test_and_filter(self, populated):
        query = [1.0, 0.0] + encode_single(TemporalInterval.of_year(2021)).to_vector()
        results = populated.search(
            query, top_k=10,
            filter_dict={"$and": [{"type": "10-K"}, {"year": 2023}]},
        )
        assert {doc.doc_id for doc, _ in results} == {"c"}

    def test_or_filter(self, populated):
        query = [1.0, 0.0] + encode_single(TemporalInterval.of_year(2021)).to_vector()
        results = populated.search(
            query, top_k=10,
            filter_dict={"$or": [{"year": 2021}, {"year": 2022}]},
        )
        assert {doc.doc_id for doc, _ in results} == {"a", "b"}


# ---------------------------------------------------------------------------
# Hierarchy guarding
# ---------------------------------------------------------------------------


class TestHierarchyGuard:
    def test_a_store_adopts_a_hierarchy_when_it_has_none(self):
        store = InMemoryVectorStore()
        store.set_hierarchy(DEFAULT_HIERARCHY)
        assert store.hierarchy is DEFAULT_HIERARCHY

    def test_setting_a_compatible_extension_is_allowed(self):
        store = InMemoryVectorStore(hierarchy=DEFAULT_HIERARCHY)
        store.set_hierarchy(DEFAULT_HIERARCHY.extended(MILLENNIUM_SCALE))

    def test_mixing_epochs_is_refused(self):
        """
        Phases from different epochs are not comparable. Silently accepting the mix
        would produce results that look plausible and are wrong.
        """
        store = InMemoryVectorStore(hierarchy=DEFAULT_HIERARCHY)
        with pytest.raises(ValueError):
            store.set_hierarchy(TemporalHierarchy(epoch=utc(2010, 1, 1)))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_a_new_store_is_empty(self, memory_store):
        assert memory_store.count() == 0
        assert memory_store.count_groups() == 0
        assert memory_store.get_document("missing") is None

    def test_clear_empties_the_store(self, memory_store):
        memory_store.add_documents([make_doc("a", TemporalInterval.of_year(2021))])
        memory_store.clear()
        assert memory_store.count() == 0
        assert memory_store.count_groups() == 0

    def test_adding_the_same_id_twice_overwrites(self, memory_store):
        memory_store.add_documents([make_doc("a", TemporalInterval.of_year(2021))])
        memory_store.add_documents([make_doc("a", TemporalInterval.of_year(2022))])
        assert memory_store.count() == 1
        assert memory_store.get_document("a").interval.start.year == 2022


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestEncodingSerialisation:
    def test_json_round_trip_of_a_split_representation(self):
        import json

        original = encode(TemporalInterval.spanning(2020, 2022), group_id="g")[1]
        restored = TemporalEncoding.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored.to_vector() == pytest.approx(original.to_vector())
        assert restored.representation_index == 1
        assert restored.representation_count == 3
        assert restored.group_id == "g"
        assert restored.source_interval.start == original.source_interval.start
