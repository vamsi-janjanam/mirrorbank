"""Tests for the Stage 5 release bundler."""

from __future__ import annotations

import polars as pl

from mirrorbank.evaluate.gauntlet import run_gauntlet
from mirrorbank.instruments.registry import get_schema
from mirrorbank.release.bundler import ReleaseBundle, bundle_release
from mirrorbank.sample_data import sample_dataset


def test_bundle_release_with_report(tmp_path):
    synth = sample_dataset("ach")
    schema = get_schema("ach")
    report = run_gauntlet(synth, synth, schema=schema)

    bundle = bundle_release(
        synth,
        instrument="ach",
        epsilon=3.0,
        delta=1e-5,
        out_dir=str(tmp_path),
        report=report,
    )

    assert isinstance(bundle, ReleaseBundle)

    for path in (bundle.csv_path, bundle.certificate_path, bundle.scorecard_path):
        assert path
        with open(path, "rb") as f:
            content = f.read()
        assert len(content) > 0

    read_back = pl.read_csv(bundle.csv_path)
    assert read_back.height == synth.height

    cert_html = open(bundle.certificate_path).read()
    assert "Privacy Certificate" in cert_html
    assert "3.0" in cert_html or "3" in cert_html

    scorecard_html = open(bundle.scorecard_path).read()
    assert "PASS" in scorecard_html or "FAIL" in scorecard_html


def test_bundle_release_without_report(tmp_path):
    synth = sample_dataset("ach")

    bundle = bundle_release(
        synth,
        instrument="ach",
        epsilon=3.0,
        delta=1e-5,
        out_dir=str(tmp_path),
        report=None,
    )

    scorecard_html = open(bundle.scorecard_path).read()
    assert "PASS" in scorecard_html or "FAIL" in scorecard_html
    assert "No evaluation gauntlet was run" in scorecard_html
