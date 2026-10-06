# Test Specification

## Test Strategy

The suite is split by what each layer has to guarantee:

| Area | File | Tests |
|------|------|-------|
| Caller-data contract | `tests/unit/test_caller_contract.py` | 84 |
| Retrieval + scope screen | `tests/unit/test_retrieval.py` | 26 |
| Inner workflow nodes + state | `tests/unit/test_log_c2_029.py` | 40 |
| Output boundary | `tests/unit/test_output_boundary.py` | 30 |
| End-to-end through `POST /invoke` | `tests/proof_of_boundary/test_invoke_e2e.py` | 44 |
| Domain boundary proofs | `tests/proof_of_boundary/test_pb_log_c2_029.py` | 13 |
| Framework contract | `tests/proof_of_boundary/test_pb_invoke_order.py`, `tests/unit/test_framework_compliance_tc06_tc07.py` | 6 |
| Structure guards | `tests/proof_of_boundary/test_import_isolation.py`, `test_state_safety.py`, `tests/unit/test_main_node.py` | 4 |
| Interrupt propagation (skip stub) | `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | 1 |

Everything runs offline: retrieval and answer composition are deterministic, and the
HTTP boundary is exercised through the app's real ASGI interface rather than a test
client, so no test can silently skip for a missing dependency.

```bash
python -m pytest tests/ -v
```

## Framework Contract Tests

| ID | Test | Expected result | Where |
|----|------|-----------------|-------|
| FC-01 | State is a flat TypedDict — no Pydantic, no dataclass | Structure check passes | `test_log_c2_029.py::TestState`, `test_state_safety.py` |
| FC-02 | No credential material in State | Scan finds nothing | `test_state_safety.py` |
| FC-02b | Raw ingress absent from checkpoint surfaces (PB-5) | **Auto-waived — checkpointing disabled** (`config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`); gate and traversal helper ship with the stub | `test_state_safety.py` |
| FC-03 | The default node input gate cannot be overridden | Class definition raises | `test_framework_compliance_tc06_tc07.py` |
| FC-04 | The default node output gate cannot be overridden | Class definition raises | `test_framework_compliance_tc06_tc07.py` |
| FC-05 | `required_trust_level` is enforced before `execute()` | Under-privileged caller refused, nothing executed | `test_pb_invoke_order.py` |
| FC-06 | Invoke order holds for every node under `src/nodes/` | trust gate → node_start → input gate → execute → output gate → node_complete | `test_pb_invoke_order.py` |
| FC-07 | The agent's declared minimum caller trust is enforced at its entry node | Anonymous refused, verified-external admitted | `test_pb_invoke_order.py::TestEntryTrustBoundary` |
| FC-08 | No import of the hosting platform's SDK | AST scan: 0 violations | `test_import_isolation.py` |

## Boundary Proofs

| ID | Boundary | Test | Expected result |
|----|----------|------|-----------------|
| B-1 | Scope | A question needing live shipment data | Declined with redirect text; no sources block, no policy claim |
| B-2 | Scope, other direction | A policy question using the same vocabulary ("tracking number format") | Answered normally |
| B-3 | Grounding | Every answer emitted | Carries its sources and the standing scope note |
| B-4 | Grounding | Retrieved passages that name no source | Error; no uncited answer |
| B-5 | Grounding | An answer that reached the boundary without sources | Withheld |
| B-6 | Content | Credential, national-ID or payment-card forms anywhere in the payload | Whole answer withheld; the value never appears in the response |
| B-7 | Content | The caller's own question | Never echoed back into the answer |
| B-8 | Identifiers | Route, port, container and consignment references | Byte-identical on the way out |

B-1 to B-7 are in `tests/proof_of_boundary/test_pb_log_c2_029.py` and
`tests/proof_of_boundary/test_invoke_e2e.py`; B-8 is pinned in both
`test_invoke_e2e.py` and `tests/unit/test_output_boundary.py`.

## Caller-Contract Tests

| ID | Test | Input | Expected result |
|----|------|-------|-----------------|
| C-01 | Non-finite numbers, per field | `"NaN"`, `"Infinity"`, `"-Infinity"`, raw `nan`, raw `inf` on `top_k`, `score_threshold`, `max_answer_chars` | Refused, field named, value not echoed |
| C-02 | Out-of-range numbers | `top_k: 0`, `top_k: 21`, `score_threshold: 1.01`, `max_answer_chars: 499` | Refused |
| C-03 | Free text in an identifier field | `carrier: "<script>alert(1)</script>"` | Refused; the value never appears in the error |
| C-04 | Malformed route code | `"TOKYO-OSAKA"`, `"HND OSA"` | Refused |
| C-05 | Malformed documents | Missing `document_id` / `source` / `text`, bad `last_verified`, non-object entry | Refused, entry position named |
| C-06 | Structural caps | 21 documents, over-long document text, over-long question | Refused |
| C-07 | Duplicate document ids | Two entries with the same `document_id` | Refused |
| C-08 | Instruction-override content | In the question, and in a document body | Refused by the template itself, `execute()` called directly |
| C-09 | Ordinary logistics questions containing the same words | "Which previous instructions apply to fragile cartons?" | Accepted |
| C-10 | Control characters | In the question and in document text | Stripped; paragraph structure preserved |
| C-11 | Adapter limits | Missing or wrong bearer token; oversized `input_context` | 401; 413 |

## Business Logic Tests

| ID | Test | Input | Expected result |
|----|------|-------|-----------------|
| BL-01 | A policy question is answered from the caller's own corpus | "What are the tender rules on route HND-OSA?" with `policy_documents` | Answer quotes the caller's text and cites the caller's source |
| BL-02 | With no documents sent, the bundled corpus answers | "What are the cold chain handling requirements?" | Answer cites `logistics_policy/cold_chain_handling.md` |
| BL-03 | `top_k` reaches the inner workflow | Same question with `top_k: 1` | Exactly one source cited |
| BL-04 | Relevance ranking | Multi-document corpus | Results ordered by score, deterministic across runs |
| BL-05 | Nothing relevant | "Which sonata did Beethoven publish in 1802?" | Declined, not answered from the nearest passage |
| BL-06 | Length ceiling | `max_answer_chars` with a long corpus | Body trimmed, sources and scope note kept |

## Marketplace Entry Point — `tests/unit/test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `LogisticsPolicyQAAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.
