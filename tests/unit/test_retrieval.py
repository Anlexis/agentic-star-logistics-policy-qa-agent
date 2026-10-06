# LOG-C2-029 — the retrieval service and the scope screen (src/services/service.py).
#
# The pipeline's answers are only as honest as this layer: the passages it
# returns are what the answer quotes and cites. These tests hold it to real
# behaviour — a relevant passage is found and ranked, an irrelevant corpus
# returns nothing rather than a nearest guess, and the caller's controls
# (top_k, threshold, scoping tokens) actually change the result.

import pytest

from src.services.service import BUNDLED_CORPUS, PolicyCorpusService, is_live_data_request

_DOCS = [
    {
        "document_id": "cold_chain",
        "title": "Cold chain handling",
        "source": "carrier_policy/cold_chain.md",
        "text": (
            "Cold chain consignments travel between 2 and 8 degrees Celsius.\n\n"
            "A temperature logger rides inside the packaging."
        ),
        "last_verified": "2026-05-01",
    },
    {
        "document_id": "hazardous",
        "title": "Hazardous goods",
        "source": "carrier_policy/hazardous.md",
        "text": "Hazardous consignments need a signed shipper declaration quoting the UN number.",
        "last_verified": "2026-04-01",
    },
]


def _search(query, **kwargs):
    kwargs.setdefault("score_threshold", 0.35)
    return PolicyCorpusService().search(query=query, **kwargs)


class TestRelevance:
    def test_the_matching_document_is_returned_with_its_metadata(self):
        [chunk] = _search("what temperature do cold chain consignments travel at", documents=_DOCS)
        assert chunk["source"] == "carrier_policy/cold_chain.md"
        assert chunk["document_id"] == "cold_chain"
        assert chunk["last_verified"] == "2026-05-01"
        assert 0.0 < chunk["score"] <= 1.0

    def test_the_best_passage_of_a_document_is_selected(self):
        [chunk] = _search("what rides inside the packaging", documents=_DOCS)
        assert "temperature logger" in chunk["text"]
        assert "degrees Celsius" not in chunk["text"]

    def test_an_unrelated_question_returns_nothing(self):
        assert _search("which sonata did beethoven publish in 1802", documents=_DOCS) == []

    def test_results_are_ranked_by_score(self):
        chunks = _search("hazardous consignments packaging temperature", documents=_DOCS, score_threshold=0.0)
        scores = [chunk["score"] for chunk in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_plural_and_singular_forms_match(self):
        assert _search("hazardous goods declaration", documents=_DOCS)
        assert _search("hazardous good declarations", documents=_DOCS)

    def test_the_same_query_returns_the_same_result(self):
        first = _search("cold chain temperature", documents=_DOCS)
        second = _search("cold chain temperature", documents=_DOCS)
        assert first == second


class TestCallerControls:
    def test_top_k_bounds_the_result_count(self):
        query = "hazardous consignments packaging temperature"
        assert len(_search(query, documents=_DOCS, score_threshold=0.0)) == 2
        assert len(_search(query, documents=_DOCS, score_threshold=0.0, top_k=1)) == 1

    def test_a_high_threshold_excludes_weak_matches(self):
        query = "hazardous consignments packaging temperature"
        assert _search(query, documents=_DOCS, score_threshold=0.0)
        assert _search(query, documents=_DOCS, score_threshold=0.99) == []

    def test_a_scoping_token_boosts_the_document_that_mentions_it(self):
        query = "consignments declaration"
        plain = _search(query, documents=_DOCS, score_threshold=0.0)
        boosted = _search(query, documents=_DOCS, score_threshold=0.0, context={"shipment_class": "cold_chain"})
        by_id = {chunk["document_id"]: chunk["score"] for chunk in plain}
        boosted_by_id = {chunk["document_id"]: chunk["score"] for chunk in boosted}
        assert boosted_by_id["cold_chain"] > by_id["cold_chain"]

    def test_scores_never_exceed_one(self):
        chunks = _search(
            "cold chain consignments travel between 2 and 8 degrees celsius",
            documents=_DOCS,
            score_threshold=0.0,
            context={"shipment_class": "cold_chain", "carrier": "cold", "route": "cold"},
        )
        assert all(chunk["score"] <= 1.0 for chunk in chunks)


class TestBundledCorpus:
    def test_the_bundled_corpus_is_searched_when_no_documents_are_supplied(self):
        chunks = _search("what are the cold chain handling requirements")
        assert chunks and chunks[0]["source"].startswith("logistics_policy/")

    def test_every_bundled_document_is_well_formed(self):
        for document in BUNDLED_CORPUS:
            assert document["document_id"] and document["source"] and document["text"].strip()
            assert document["last_verified"]

    def test_an_empty_document_is_skipped_not_returned(self):
        assert _search("packaging", documents=[{"document_id": "x", "source": "s.md", "text": "  "}]) == []


class TestScopeScreen:
    @pytest.mark.parametrize(
        "text",
        [
            "what is the eta for this shipment",
            "give me the current status of my order",
            "I need real-time visibility",
            "live tracking for the consignment please",
            "where is my parcel",
            "track my shipment",
        ],
    )
    def test_live_data_questions_are_recognised(self, text):
        assert is_live_data_request(text)

    @pytest.mark.parametrize(
        "text",
        [
            "what format does a tracking number use",
            "please give a detailed answer on packaging",
            "which delivery standards apply to fragile goods",
            "what is the retail packaging rule",
            "how is a beta programme consignment labelled",
            "what does the policy say about live animals in cold chain",
        ],
    )
    def test_ordinary_policy_questions_are_not_scoped_out(self, text):
        """An unanchored screen matched 'eta' inside 'detailed' and 'live'
        inside 'delivery' — and refused genuine policy work."""
        assert not is_live_data_request(text)

    def test_empty_text_is_not_a_live_data_request(self):
        assert not is_live_data_request("")
