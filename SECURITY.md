# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

This is a personal research/portfolio project, not a production service.
Still, reports are welcome:

- Open a GitHub issue (for anything that does not expose sensitive data), or
- Email the maintainer: <maintainer-email@example.com> <!-- placeholder -->

You can expect a response on a best-effort basis.

## Scope and threat model

Mirrorbank generates **synthetic** financial data under differential privacy.
A few things worth being explicit about:

- **The DP guarantees apply to the generation pipeline only** (DP-SGD
  training charged against a `PrivacyBudget`). They say nothing about how
  you handle the real input dataset before or after generation.
- **Real input data must never be committed to this repository.** The
  `.gitignore` excludes `data/raw/`, `data/synthetic/`, `outputs/`, and
  `.env*` for this reason — keep sensitive datasets in those directories.
  Only `data/sample/` (fake, generated examples) is committed.
- All identifiers produced by `src/mirrorbank/reference/` (routing numbers,
  SWIFT/BIC, IMAD, MICR) are syntactically valid but entirely fake.
- Outputs are research artifacts. Do not treat a privacy certificate from
  this tool as a substitute for a formal privacy review.

## Out of scope

- Vulnerabilities in upstream dependencies (report those upstream; we run
  `pip-audit` in CI to catch known CVEs).
- Attacks requiring access to the machine running the pipeline.
