# Logistics Policy Q&A Agent

AI agent for answering logistics policy questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Logistics
> **Template ID**: LOG-C2-029

## Overview

Answers logistics policy questions from a corpus of policy text, and cites what it
answered from. Send a question — packaging standards, cold chain handling, hazardous
goods acceptance, route service levels, damage and loss claims — and the agent finds
the relevant passages, composes an answer out of them, and prints the sources and the
date those sources were last verified underneath it.

The corpus is yours: send your own policy documents with the request and the answer is
drawn from those. With no documents supplied the agent answers from the small reference
corpus it ships with, so a fresh checkout does real work before anything is wired in.

Retrieval and composition are both deterministic — a lexical relevance search followed
by fixed templating — so the same question over the same corpus always produces the same
answer and the same citations, and the pipeline runs end to end with no model or external
service. `src/services/service.py` is where a vector or hybrid search client goes when you
outgrow that.

Two boundaries are worth knowing about before you adapt it. The agent declines questions
that need live shipment data — current status, location, estimated arrival — rather than
guessing at them from policy text. And every answer that leaves the agent must carry its
sources and a standing note about what static policy text can and cannot tell you; the
output boundary enforces both independently of how the answer was composed, and withholds
an answer it cannot vouch for.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at
graph compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Using it

`POST /invoke` takes the question in `input`, plus an optional `input_context` object of
per-invocation options:

```json
{
  "input": "What temperature range applies to cold chain consignments?",
  "input_context": {
    "shipment_class": "cold_chain",
    "top_k": 3,
    "policy_documents": [
      {
        "document_id": "cold_chain_handling",
        "title": "Cold chain handling",
        "source": "carrier_policy/cold_chain.md",
        "text": "Cold chain consignments travel between 2 and 8 degrees Celsius.",
        "last_verified": "2026-05-04"
      }
    ]
  }
}
```

Every caller-supplied number is parsed by a finite, bounded parser and every caller-supplied
string that selects behaviour is locked to an inert identifier; a violation fails closed with
an error naming the field. `docs/02_design.md` documents the full contract, the bounds and the
output schema.

## Project Structure

```
src/          agent implementation (nodes, graphs, services, schemas, HTTP adapter)
src/examples/ annotated samples of the framework's agent patterns
tests/        unit and boundary tests
config/       agent manifest + runtime configuration
docs/         design and test documentation
```

See `docs/` for the design specification and test specification.

## Customising

1. Adjust `config/` for your own environment and retrieval defaults.
2. Send your own `policy_documents`, or replace the reference corpus in
   `src/services/service.py` — that module also holds the relevance search, which is the
   first thing to swap for a real vector or hybrid client.
3. Review the node implementations under `src/nodes/` — the caller contract bounds, the
   scope screen and the output contract are the usual things to localise.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
