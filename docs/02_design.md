# Template Design Specification

## Position in the framework

- **Agent class**: `LogisticsPolicyQAAgent` (`src/graph/graph.py`)
- **Category**: Cat 2 — a domain-specific pipeline behind the fixed agent backbone
- **Composition**: two layers. The outer graph is the framework's five-node backbone;
  its `main` slot holds a `GraphNode` that runs an inner workflow graph.

| Layer | Class |
|-------|-------|
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Outer graph | `LogisticsPolicyQAAgent` |
| `main` slot wrapper | `LogisticsPolicyQAGraphNode` |
| Inner graph | `DomainWorkflowGraph` (`BaseGraph`, fully custom topology) |
| State | `State` — flat TypedDict extending the framework's `AgentState` |

Three-layer separation:

- **State**: flat TypedDict composition. Compound values are stored JSON-serialised so
  everything reaching a checkpoint is a msgpack-safe primitive. No Pydantic models —
  they corrupt silently under checkpoint serialisation.
- **Node**: `execute(self, state) -> dict`, returning only the keys the node changed.
- **Graph**: composition through `register_nodes()`; backbone wiring is never overridden.

## Architecture

### Outer backbone

```
START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
                                        |
                                        +-- retry -> pre_process
```

| Slot | Class | Responsibility | Reads | Writes |
|------|-------|----------------|-------|--------|
| initialize | framework default | session, schema version, caller trust | — | session_id, schema_version, trust_level |
| pre_process | `PreProcessNode` | the caller-data contract | user_input, input_context | validated_input, validated_context, enriched_context |
| main | `LogisticsPolicyQAGraphNode` | runs the inner workflow | validated_input, validated_context | answer, result, citations, validation flags |
| post_process | `PostProcessNode` | the external-output boundary | validated_answer, citations, caller text | formatted_output |
| finalize | framework default | response metadata, timing | — | response_metadata, total_time_ms |

### Inner workflow

```
START -> query_normalize -> kb_retrieve -> answer_generate -> response_validate -> END
```

| Node | Responsibility | Reads | Writes |
|------|----------------|-------|--------|
| `QueryNormalizeNode` | normalise the question; resolve carrier / route / shipment class | user_input, validated_context | normalized_query, extracted_context |
| `LogisticsPolicyKBRetrieveNode` | relevance search over the corpus | normalized_query, extracted_context, validated_context | retrieved_chunks, out_of_scope, out_of_scope_reason |
| `AnswerGenerateNode` | compose the cited answer | retrieved_chunks, validated_context | answer, result, citations, last_verified |
| `ResponseValidateNode` | check the composed answer | result, out_of_scope, validated_context | validated_answer, validation_passed |

### Crossing the graph boundary

The framework hands a subgraph a plain string and does not forward the outer caller
context with it. The validated caller options therefore travel explicitly:
`LogisticsPolicyQAGraphNode.extract_input()` stashes them
(`src/graph/context_bridge.py`) and `DomainWorkflowGraph._extra_initial_state()` reads
them back into the inner initial state. Only PreProcessNode's validated output crosses,
never the raw request, so the inner workflow reads bounds-checked values by construction.

A `ContextVar` carries the hand-off, so concurrent invocations in one process cannot see
each other's context.

## Configuration

Two files with distinct jobs:

- `config/agent.yaml` — the static manifest read by the agent registry. Flat: every key at
  root level, no `agent:` block, and `class:` is a single dotted import path. It declares
  the minimum caller trust level and the compile-time `requires` (both empty here — the
  template constructs no client and requires no secret).
- `config/config.yaml` — the runtime block (`max_retry`, `timeout_s`) passed to the graph
  constructor.

`LogisticsPolicyQAAgent.__init__` loads the runtime block when no config is passed, and
`LogisticsPolicyQAGraphNode._parent_config()` forwards the declared values to the inner
graph, which is constructed separately and would otherwise be built with no config at
all. The inner graph validates any value that IS declared, so a deployment mistake fails
at compile time rather than mid-run. A declared value that reached neither graph would be
worse than no value at all: the file would say one thing and the run would do another,
silently.

Retrieval behaviour (`top_k`, `score_threshold`, `max_answer_chars`) is per invocation
rather than per deployment; the defaults live in `src/schemas/state.py` alongside the
bounds that constrain them.

## Caller-data contract

`POST /invoke` accepts the question in `input` and options in `input_context`. The HTTP
adapter refuses an `input_context` over 256 KB before the agent is entered at all; every
field below is then validated by `PreProcessNode`, which fails CLOSED and names the
offending field without ever echoing its value.

| Field | Type | Bounds |
|-------|------|--------|
| `input` (the question) | string | non-empty, ≤ 2,000 characters, control characters stripped, instruction-override content refused |
| `policy_documents` | list of objects | ≤ 20 entries, unique `document_id` |
| `policy_documents[].document_id` | string | required, `[a-z0-9_]{1,32}` |
| `policy_documents[].source` | string | required, citation label ≤ 120 characters |
| `policy_documents[].text` | string | required, ≤ 8,000 characters, control characters stripped |
| `policy_documents[].title` | string | optional, ≤ 200 characters |
| `policy_documents[].last_verified` | string | optional, `YYYY-MM-DD` |
| `carrier`, `shipment_class`, `channel` | string | `[a-z0-9_]{1,32}` |
| `route_code` | string | `AAA-BBB` (three uppercase letters either side) |
| `top_k` | integer | 1–20, default 5 |
| `score_threshold` | number | 0.0–1.0, default 0.35 |
| `max_answer_chars` | integer | 500–20,000, default 10,000 |

Two rules behind that table are worth stating explicitly, because they are what keeps
the contract closed rather than merely documented:

- **Every caller-controlled number** goes through one finite, bounded parser
  (`finite_in_range` in `src/schemas/state.py`). `"NaN"` and `"Infinity"` parse
  through `float()` and arrive intact through raw JSON, and every comparison against
  NaN is False — so an unchecked value silently disables the control it was meant to
  set. Anything non-finite or out of range is refused.
- **Every caller string that selects behaviour** — carrier, shipment class, channel,
  document id — is locked to an inert identifier. Free text in such a field is
  caller-controlled content on an output surface.

Document titles and bodies are genuinely free text and are handled as such: length
capped, control characters stripped, and refused outright if they carry an
instruction-override directive.

## Security design

| Concern | Where it is enforced |
|---------|---------------------|
| Caller trust | The agent declares a minimum caller trust level in `config/agent.yaml`; the entry node enforces it, and the HTTP adapter establishes it from a bearer token when no upstream middleware has. |
| Input validation | `PreProcessNode`, fail-closed, field-naming errors, no value echoed. |
| Instruction-override refusal | `PreProcessNode`, inside `execute()`. The platform screens hostile input ahead of `execute()` too, but a template whose only defence is the platform's answers normally wherever that screen is absent or off — so the refusal is the template's own and is proved by calling `execute()` directly. |
| Output content | `PostProcessNode` and `ResponseValidateNode`, sharing one pattern definition in `src/services/service.py` and applying it independently. |
| Audit | Each node emits a domain event through `emit_trace_event()` inside `execute()`; the framework emits the lifecycle events itself and templates must not duplicate them. Event payloads carry counts and reason codes, never caller text. |

**Completion is not the same as answering.** A run that ends with
`AgentStatus.SUCCESS` reports that the request was handled safely to a defined
end, not that the request was carried out. A value the caller can correct (an
out-of-contract parameter, an empty or over-long request) ends this way so the
caller receives the reason and can send a corrected request on the same
conversation; terminating instead would end the calling surface's turn and
surface only an exception type, leaving the reason reachable solely from the
audit trail. The reason travels as `error_code` in State, every later domain
node passes through without doing work once it is set, the structured output
fields are withheld, and the output boundary renders the reason as a static
caller-facing sentence.

Two classes keep terminating, and must not be folded into the above: content
the agent refuses outright (an instruction-override payload — re-sending a
reworded variant is not a correction), and a breach of a contract the caller
cannot influence.

## Output contract

Every answer that leaves the agent:

1. is composed from retrieved policy passages, never from the caller's own words —
   the question is not echoed back, and a verbatim reappearance of it as a whole
   phrase is redacted at the boundary;
2. carries a **Sources** block naming at least one source, and the date the most
   recently verified source was last checked;
3. carries the standing scope note stating that the answer reflects static policy text
   and not the live status of any shipment;
4. contains no credential, national-ID or payment-card form.

`PostProcessNode` enforces all four independently of how the answer was composed, in
that order, and re-runs the content scan on the final bytes. An answer that lost its
sources is withheld — citations cannot be invented. An answer that lost only the scope
note has it restored, with an audit event. A scope redirect makes no policy claim and
cites nothing; that path is recognised from the workflow's own flag, never guessed at
from the text.

### On numeric rounding grids

Some templates in this family render monetary aggregates and snap every monetary token
onto a rounding grid on the way out. **That grid is deliberately not used here.** This
agent emits no monetary aggregate — it answers policy questions in the words of the
policy — and the grammar such a grid uses treats any standalone three-letter uppercase
word as a currency marker. In this domain those words are the content: route and port
codes (`HND-OSA`), carrier codes, container and consignment references. A grid here
would rewrite the identifiers the answer exists to quote. The invariant enforced instead
is the one above — cited, scope-marked, free of restricted content — and the boundary
tests pin that this domain's identifier forms pass through byte-identical.

## Scope boundary

A static policy corpus can say what the rules are; it cannot say where a parcel is now.
Questions asking for current status, location or estimated arrival are declined with
redirect text rather than answered from the nearest-matching policy passage.

The screen behind that decision is anchored on whole phrases and whole words, and is
tested in both directions. An unanchored version matched `eta` inside "detailed" and
`live` inside "delivery", which declined ordinary policy questions — the failure
direction that blocks real work. The standing scope note itself names live status by
design, so the answer-side backstop screens only the answer body, never the footer.

## Retrieval design

`PolicyCorpusService` (`src/services/service.py`) scores each passage by lexical
coverage — the share of the question's content words the passage contains, with regular
plurals folded and function words dropped — plus fixed boosts when the passage also
mentions the carrier, route or shipment class resolved for the request. Passages at or
above the threshold are ranked and the top `top_k` returned.

It is deterministic: the same question over the same corpus always returns the same
passages in the same order, so an answer can be audited after the fact. Swap the
`search()` body for a vector or hybrid client to answer from a larger corpus; the return
shape is what the rest of the pipeline depends on.

With no `policy_documents` supplied, the small reference corpus in the same module is
searched instead, so a fresh checkout answers real questions before anything is wired in.

## State constraints

- Flat TypedDict only — primitives and JSON-serialised strings.
- No credentials or tokens in State (they would reach the checkpoint store).
- Invocation context travels through `config["configurable"]`, not through State.
- No Pydantic models, dataclasses or arbitrary Python objects.
- Fields inherited from the framework state are not re-declared.

## Design decisions

| Decision | Alternative | Chosen | Rationale |
|----------|-------------|--------|-----------|
| Base class | AutonomousBaseGraph | AgentBaseGraph | Fixed pipeline, not an autonomous loop |
| Composition | single node in the `main` slot | inner graph behind a `GraphNode` | Four distinct steps with their own state; keeps the outer backbone uniform |
| Answer composition | model-generated prose | deterministic templating over retrieved passages | The answer can be checked against its citations line by line, and the template runs with no model or external service |
| Corpus | fixed built-in only | caller-supplied, with a bundled fallback | The corpus is the customer's; the fallback keeps a fresh checkout useful |
| Out-of-scope representation | error status | `out_of_scope=True` with a success status | The run behaved correctly; the question simply was not answerable from policy text |
| Runtime config | leave the graph on framework defaults | load `config/config.yaml` and forward it inward | A declared value that governs nothing is worse than no value — the file and the run must agree |
| Inner-node trust level | re-gate at every node | gate once at the entry boundary | The inner workflow is unreachable except through the entry node; re-gating at a higher level there would refuse the very callers the agent is published for |

## Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Caller-auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.
