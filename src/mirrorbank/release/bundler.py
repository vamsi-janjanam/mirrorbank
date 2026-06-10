"""Stage 5 — Release bundling.

Bundles a synthetic dataset, a privacy certificate (HTML), and a utility
scorecard (HTML) into a versioned release directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl
from jinja2 import Template

from mirrorbank.evaluate.gauntlet import GauntletReport

_CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem;
       color: #1a1a2e; background: #f7f7fb; }
.card { background: #fff; border-radius: 12px; padding: 1.5rem 2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); max-width: 720px; margin: 0 auto; }
h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
.subtitle { color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee;
         font-size: 0.95rem; }
th { color: #555; font-weight: 600; width: 45%; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
         font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; }
.pass { background: #e6f7ed; color: #1a7f37; }
.fail { background: #fdecec; color: #c0392b; }
.footer { margin-top: 1.5rem; font-size: 0.8rem; color: #999; }
"""

_CERTIFICATE_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mirrorbank Privacy Certificate</title>
<style>{{ css }}</style>
</head>
<body>
<div class="card">
  <h1>Mirrorbank Privacy Certificate</h1>
  <div class="subtitle">
    Instrument: {{ instrument }} &middot; Version: {{ version }} &middot;
    Generated: {{ generated_at }}
  </div>
  <table>
    <tr><th>Instrument</th><td>{{ instrument }}</td></tr>
    <tr><th>Release version</th><td>{{ version }}</td></tr>
    <tr><th>Generated at (UTC)</th><td>{{ generated_at }}</td></tr>
    <tr><th>Synthetic row count</th><td>{{ row_count }}</td></tr>
    <tr><th>Differential privacy guarantee</th>
        <td>(&epsilon; = {{ epsilon }}, &delta; = {{ delta }})-DP</td></tr>
    {% if mia_auc is not none %}
    <tr><th>Membership inference attack AUC</th>
        <td>{{ "%.3f"|format(mia_auc) }}
        <span class="badge {{ 'pass' if mia_pass else 'fail' }}">
          {{ 'PASS' if mia_pass else 'FAIL' }}
        </span></td></tr>
    {% endif %}
  </table>
  <div class="footer">
    This certificate confirms that the accompanying synthetic dataset was
    generated under the formal differential privacy guarantee stated above.
    No further evaluation is implied for fields not listed.
  </div>
</div>
</body>
</html>
"""
)

_SCORECARD_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Mirrorbank Utility Scorecard</title>
<style>{{ css }}</style>
</head>
<body>
<div class="card">
  <h1>Mirrorbank Utility Scorecard</h1>
  <div class="subtitle">
    Instrument: {{ instrument }} &middot; Version: {{ version }} &middot;
    Generated: {{ generated_at }}
  </div>
  {% if has_report %}
  <table>
    <tr><th>Overall result</th>
        <td><span class="badge {{ 'pass' if pass_overall else 'fail' }}">
          {{ 'PASS' if pass_overall else 'FAIL' }}
        </span></td></tr>
    <tr><th>Fidelity &mdash; KS pass rate</th>
        <td>{{ "%.1f"|format(ks_pass_rate * 100) }}%</td></tr>
    <tr><th>Fidelity &mdash; correlation distance</th>
        <td>{{ "%.3f"|format(corr_distance) }}</td></tr>
    <tr><th>Fidelity &mdash; instrument rule errors</th>
        <td>{{ instrument_error_count }}</td></tr>
    {% if utility_ratio is not none %}
    <tr><th>Utility &mdash; AUC_TSTR / AUC_TRTR</th>
        <td>{{ "%.3f"|format(utility_ratio) }} (target &ge; 0.90)
        <span class="badge {{ 'pass' if utility_pass else 'fail' }}">
          {{ 'PASS' if utility_pass else 'FAIL' }}
        </span></td></tr>
    {% endif %}
    {% if mia_auc is not none %}
    <tr><th>Privacy &mdash; MIA attack AUC</th>
        <td>{{ "%.3f"|format(mia_auc) }} (target &le; 0.55)
        <span class="badge {{ 'pass' if mia_pass else 'fail' }}">
          {{ 'PASS' if mia_pass else 'FAIL' }}
        </span></td></tr>
    {% endif %}
  </table>
  {% else %}
  <p>No evaluation gauntlet was run for this release. Generate a
  <code>GauntletReport</code> and pass it to <code>bundle_release</code>
  to populate this scorecard.</p>
  <p><span class="badge fail">FAIL</span> No results to report.</p>
  {% endif %}
  <div class="footer">
    Synthetic row count: {{ row_count }}
  </div>
</div>
</body>
</html>
"""
)


@dataclass
class ReleaseBundle:
    out_dir: str
    csv_path: str
    certificate_path: str
    scorecard_path: str
    version: str


def bundle_release(
    synthetic: pl.DataFrame,
    *,
    instrument: str,
    epsilon: float,
    delta: float,
    out_dir: str,
    report: GauntletReport | None = None,
    version: str | None = None,
) -> ReleaseBundle:
    """Bundle a synthetic dataset, privacy certificate, and utility scorecard."""
    now = datetime.now(UTC)
    if version is None:
        version = now.strftime("v%Y%m%d-%H%M%S")
    generated_at = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "synthetic_transactions.csv")
    synthetic.write_csv(csv_path)

    row_count = synthetic.height

    mia_auc = report.privacy.attack_auc if report and report.privacy else None
    mia_pass = report.privacy.pass_privacy if report and report.privacy else None

    certificate_html = _CERTIFICATE_TEMPLATE.render(
        css=_CSS,
        instrument=instrument,
        version=version,
        generated_at=generated_at,
        row_count=row_count,
        epsilon=epsilon,
        delta=delta,
        mia_auc=mia_auc,
        mia_pass=mia_pass,
    )
    certificate_path = os.path.join(out_dir, "privacy_certificate.html")
    with open(certificate_path, "w") as f:
        f.write(certificate_html)

    scorecard_html = _SCORECARD_TEMPLATE.render(
        css=_CSS,
        instrument=instrument,
        version=version,
        generated_at=generated_at,
        row_count=row_count,
        has_report=report is not None,
        pass_overall=report.pass_overall if report else None,
        ks_pass_rate=report.fidelity.ks_pass_rate if report else None,
        corr_distance=report.fidelity.corr_distance if report else None,
        instrument_error_count=len(report.fidelity.instrument_errors)
        if report
        else None,
        utility_ratio=report.utility.ratio if report and report.utility else None,
        utility_pass=report.utility.pass_utility
        if report and report.utility
        else None,
        mia_auc=mia_auc,
        mia_pass=mia_pass,
    )
    scorecard_path = os.path.join(out_dir, "utility_scorecard.html")
    with open(scorecard_path, "w") as f:
        f.write(scorecard_html)

    return ReleaseBundle(
        out_dir=os.path.abspath(out_dir),
        csv_path=os.path.abspath(csv_path),
        certificate_path=os.path.abspath(certificate_path),
        scorecard_path=os.path.abspath(scorecard_path),
        version=version,
    )
