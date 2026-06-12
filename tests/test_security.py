"""Security regression tests — HTML escaping in release bundles, CLI input validation."""

from __future__ import annotations

import argparse

import pytest

from mirrorbank.sample_data import sample_dataset

_XSS = "<script>alert(1)</script>"


@pytest.fixture
def real_csv(tmp_path):
    df = sample_dataset("ach", n=200)
    path = tmp_path / "real_ach.csv"
    df.write_csv(path)
    return path


# ── Release bundler: HTML escaping ───────────────────────────────────────────


def test_bundle_release_escapes_html_in_string_fields(tmp_path):
    from mirrorbank.release.bundler import bundle_release

    synth = sample_dataset("ach")
    bundle = bundle_release(
        synth,
        instrument=_XSS,
        epsilon=3.0,
        delta=1e-5,
        out_dir=str(tmp_path),
        report=None,
        version=f'v1"{_XSS}',
    )

    for path in (bundle.certificate_path, bundle.scorecard_path):
        html = open(path).read()
        assert _XSS not in html, f"unescaped script tag in {path}"
        assert "&lt;script&gt;" in html

    # Structure and numeric formatting are intact despite escaping.
    cert_html = open(bundle.certificate_path).read()
    assert "Privacy Certificate" in cert_html
    assert "3.0" in cert_html
    assert "1e-05" in cert_html or "1e-5" in cert_html


def test_certificate_template_escapes_directly():
    from mirrorbank.release.bundler import _CERTIFICATE_TEMPLATE

    html = _CERTIFICATE_TEMPLATE.render(
        css="",
        instrument=_XSS,
        version=_XSS,
        generated_at=_XSS,
        row_count=10,
        epsilon=3.0,
        delta=1e-5,
        mia_auc=0.512,
        mia_pass=True,
    )
    assert _XSS not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "0.512" in html  # numeric formatting still works under autoescape


# ── CLI input validation ─────────────────────────────────────────────────────


def test_cli_profile_rejects_missing_input(tmp_path):
    from mirrorbank.cli import cmd_profile

    args = argparse.Namespace(csv=str(tmp_path / "does_not_exist.csv"), instrument=None)
    with pytest.raises(SystemExit) as exc:
        cmd_profile(args)
    assert exc.value.code == 1


def test_cli_profile_rejects_directory_input(tmp_path):
    from mirrorbank.cli import cmd_profile

    args = argparse.Namespace(csv=str(tmp_path), instrument=None)
    with pytest.raises(SystemExit) as exc:
        cmd_profile(args)
    assert exc.value.code == 1


def test_cli_evaluate_rejects_missing_input(tmp_path, real_csv):
    from mirrorbank.cli import cmd_evaluate

    args = argparse.Namespace(
        real=str(real_csv),
        synth=str(tmp_path / "missing_synth.csv"),
        instrument="ach",
    )
    with pytest.raises(SystemExit) as exc:
        cmd_evaluate(args)
    assert exc.value.code == 1


def test_cli_generate_rejects_out_path_that_is_a_directory(tmp_path, real_csv):
    from mirrorbank.cli import cmd_generate

    out_dir = tmp_path / "i_am_a_directory"
    out_dir.mkdir()
    args = argparse.Namespace(
        csv=str(real_csv),
        instrument="ach",
        preset="balanced",
        rows=50,
        model="dp_statistical",
        out=str(out_dir),
        seed=0,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_generate(args)
    assert exc.value.code == 1


def test_cli_release_rejects_out_dir_that_is_a_file(tmp_path, real_csv):
    from mirrorbank.cli import cmd_release

    bad_out_dir = tmp_path / "not_a_dir.txt"
    bad_out_dir.write_text("regular file")
    args = argparse.Namespace(
        csv=str(real_csv),
        instrument="ach",
        preset="balanced",
        rows=50,
        model="dp_statistical",
        out_dir=str(bad_out_dir),
        seed=0,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_release(args)
    assert exc.value.code == 1
