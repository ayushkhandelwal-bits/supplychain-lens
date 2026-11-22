# ADR-0001 — License and reuse strategy

- **Status:** Accepted (revisit when the reference repository adds a LICENSE file)
- **Date:** 2026-10-03

## Context

The reference project [NEXUS-SCCT](https://github.com/ahmedzainsyed/nexus-scct) has no `LICENSE` file. Its README shows an "Apache 2.0" badge that links to that missing file (audit check A1).

A badge states an intention, but the license grant normally lives in the license file. Without one, the default under copyright law is that the author keeps all rights. GitHub's terms let people view and fork public repositories *on GitHub*; they don't grant the right to copy code into another project.

## Decision

1. **Do not copy code, prose or UI from NEXUS-SCCT** while the license is unconfirmed.
2. **Reimplement** from understanding. Ideas, architecture, algorithms, schemas, KPI definitions and synthetic-data distributions are not protected by copyright; specific code and text are.
3. **Attribute everything we learned from it** — in `ATTRIBUTION.md`, the README credits, and a header comment in each module whose design came from NEXUS.
4. **Ask the author** to add the LICENSE file (`docs/license_request.md`).
5. Keep a local clone in `reference/` (git-ignored) for study and for `audit/verify_findings.py`.

If the author confirms Apache-2.0, we *may* reuse specific code. In that case we must ship the Apache-2.0 text, keep the original copyright notice, add a prominent "modified by" notice to each changed file, and keep any NOTICE file. We would not use "NEXUS" as if it were our product name; Apache-2.0 grants no trademark rights.

## Consequences

- More work than forking — but every line in this repository is ours, which makes the contribution story unambiguous in interviews.
- Our own license (chosen in Phase 1) applies only to our code. Apache-2.0 is the recommended choice, to align with the original author's stated intent.

*Not legal advice. This is the conservative practice used in open-source projects.*
