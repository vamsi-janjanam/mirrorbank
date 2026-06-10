"""Tests for the generate/evaluate/release CLI commands."""

from __future__ import annotations

import argparse

import polars as pl
import pytest

from mirrorbank.sample_data import sample_dataset


@pytest.fixture
def real_csv(tmp_path):
    df = sample_dataset("ach", n=200)
    path = tmp_path / "real_ach.csv"
    df.write_csv(path)
    return path


def test_cmd_generate(tmp_path, real_csv):
    from mirrorbank.cli import cmd_generate

    out = tmp_path / "synthetic.csv"
    args = argparse.Namespace(
        csv=str(real_csv),
        instrument="ach",
        preset="balanced",
        rows=50,
        model="dp_statistical",
        out=str(out),
        seed=0,
    )
    cmd_generate(args)

    synth = pl.read_csv(out)
    assert synth.height == 50


def test_cmd_evaluate(tmp_path, real_csv):
    from mirrorbank.cli import cmd_evaluate, cmd_generate

    synth_path = tmp_path / "synthetic.csv"
    cmd_generate(argparse.Namespace(
        csv=str(real_csv),
        instrument="ach",
        preset="balanced",
        rows=50,
        model="dp_statistical",
        out=str(synth_path),
        seed=0,
    ))

    args = argparse.Namespace(
        real=str(real_csv),
        synth=str(synth_path),
        instrument="ach",
    )
    # Should run without raising.
    cmd_evaluate(args)


def test_cmd_release(tmp_path, real_csv):
    pytest.importorskip("mirrorbank.release.bundler")
    from mirrorbank.cli import cmd_release

    out_dir = tmp_path / "release"
    args = argparse.Namespace(
        csv=str(real_csv),
        instrument="ach",
        preset="balanced",
        rows=50,
        model="dp_statistical",
        out_dir=str(out_dir),
        seed=0,
    )
    cmd_release(args)

    files = list(out_dir.rglob("*"))
    assert any(f.suffix == ".csv" for f in files)
    assert len([f for f in files if f.is_file()]) >= 3
