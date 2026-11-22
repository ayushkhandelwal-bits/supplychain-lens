# ADR-0010 — Project name: SupplyChain Lens

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

The deep-learning change request proposed the name **SupplyLens**. A check before adoption found:

- **SupplyLens™ Pro** — a commercial supply-chain analytics platform by Lytica (spend analytics, supplier risk intelligence), carrying a ™ mark. Same domain.
- Two public GitHub repositories named SupplyLens.

**SupplyChain Lens** was checked next:

- No trademarked product found under that name.
- Similar names exist in non-competing uses: AWS's Well-Architected *Supply Chain Lens* (guidance documents), *SupplyChainLens.com* (a news aggregator), and three GitHub repositories.

This was a web search, not a legal clearance.

## Decision

| Where | Spelling |
|---|---|
| Display name (dashboard, README, docs, LinkedIn) | **SupplyChain Lens** |
| GitHub repository | `supplychain-lens` |
| Python package | `supplychain_lens` |
| Expanded name | *SupplyChain Lens — Deep Learning–Driven Supply Chain Resilience & Decision Intelligence* |

The README carries this disclaimer:

> Personal portfolio project. Not affiliated with SupplyLens™ (Lytica), Amazon Web Services, or SupplyChainLens.com.

## Consequences

- One consistent name everywhere; renaming later would be a search-and-replace.
- The disclaimer reduces any risk of confusion with existing products.
