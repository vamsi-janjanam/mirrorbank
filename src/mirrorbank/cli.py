"""Mirrorbank CLI — instruments, profile, and budget commands."""

from __future__ import annotations

import argparse
import sys


def cmd_instruments(_args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table

    from mirrorbank.instruments.registry import REGISTRY

    console = Console()
    table = Table(title="Supported Payment Instruments", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Display Name", style="bold")
    table.add_column("Columns", justify="right")
    table.add_column("Training Cols", justify="right")
    table.add_column("Text Cols", justify="right")
    table.add_column("Fraud Label", style="yellow")

    for key in sorted(REGISTRY):
        schema = REGISTRY[key]()
        table.add_row(
            key,
            schema.display_name,
            str(len(schema.columns)),
            str(len(schema.training_columns())),
            str(len(schema.text_columns())),
            schema.fraud_label or "—",
        )

    console.print(table)


def cmd_profile(args: argparse.Namespace) -> None:
    import polars as pl
    from rich.console import Console
    from rich.table import Table

    from mirrorbank.instruments.registry import REGISTRY, detect_instrument
    from mirrorbank.profile.schema_profiler import SchemaProfiler

    console = Console()

    try:
        df = pl.read_csv(args.csv)
    except Exception as e:
        console.print(f"[red]Error reading CSV:[/red] {e}")
        sys.exit(1)

    if args.instrument:
        if args.instrument not in REGISTRY:
            console.print(f"[red]Unknown instrument:[/red] {args.instrument!r}. Choose from: {', '.join(sorted(REGISTRY))}")
            sys.exit(1)
        schema = REGISTRY[args.instrument]()
        instrument_name = args.instrument
    else:
        try:
            instrument_name = detect_instrument(df)
            schema = REGISTRY[instrument_name]()
            console.print(f"[dim]Auto-detected instrument:[/dim] [cyan]{instrument_name}[/cyan]")
        except ValueError:
            schema = None
            instrument_name = "unknown"
            console.print("[dim]Could not auto-detect instrument — profiling without schema.[/dim]")

    profiler = SchemaProfiler(schema=schema)
    profile = profiler.fit(df)

    console.print(f"\n[bold]Dataset Profile[/bold]  ·  [cyan]{profile.instrument}[/cyan]  ·  {profile.n_rows:,} rows  ·  {profile.n_cols} columns\n")

    table = Table(show_lines=False, expand=True)
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Kind", style="bold")
    table.add_column("Dtype")
    table.add_column("Unique", justify="right")
    table.add_column("Nulls", justify="right")
    table.add_column("PII", justify="center")
    table.add_column("Stats", style="dim")

    KIND_COLOR = {
        "continuous": "green",
        "categorical": "blue",
        "datetime": "magenta",
        "free_text": "yellow",
        "identifier": "dim",
        "reference": "dim cyan",
    }

    for col in profile.columns:
        color = KIND_COLOR.get(col.kind.value, "")
        kind_str = f"[{color}]{col.kind.value}[/{color}]" if color else col.kind.value
        pii_str = "[red]✓[/red]" if col.is_pii else ""
        stats_parts = []
        if "mean" in col.stats:
            stats_parts.append(f"mean={col.stats['mean']:.2f}")
        if "min" in col.stats and "max" in col.stats:
            stats_parts.append(f"range=[{col.stats['min']:.2g}, {col.stats['max']:.2g}]")
        if "top" in col.stats:
            stats_parts.append(f"top={col.stats['top']!r}")
        table.add_row(
            col.name,
            kind_str,
            col.dtype,
            f"{col.n_unique:,}",
            f"{col.null_frac:.1%}" if col.null_frac > 0 else "0%",
            pii_str,
            "  ".join(stats_parts),
        )

    console.print(table)

    training = profile.training_columns()
    text = profile.text_columns()
    console.print(f"\n[dim]→ Track A (DP generator):[/dim] {len(training)} columns  |  [dim]Track B (vocab sampler):[/dim] {len(text)} columns")


def cmd_budget(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table

    from mirrorbank.privacy.budget import BudgetConfig

    console = Console()

    presets = ["tight", "balanced", "loose", "demo"]

    if args.preset:
        targets = [args.preset]
    else:
        targets = presets

    table = Table(title="Privacy Budget Configuration", show_lines=True)
    table.add_column("Preset", style="cyan")
    table.add_column("ε (epsilon)", justify="right", style="bold")
    table.add_column("δ (delta)", justify="right")
    table.add_column("Noise σ", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Batch", justify="right")
    table.add_column("Note")

    notes = {
        "tight": "strongest privacy — lower utility",
        "balanced": "recommended default",
        "loose": "higher utility — weaker privacy",
        "demo": "demo only — do not use in production",
    }

    for preset in targets:
        try:
            cfg = BudgetConfig.from_preset(preset, dataset_size=args.rows, batch_size=args.batch_size)
            table.add_row(
                preset,
                str(cfg.epsilon),
                f"{cfg.delta:.0e}",
                str(cfg.noise_multiplier),
                f"{args.rows:,}",
                f"{args.batch_size:,}",
                notes.get(preset, ""),
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    console.print(table)


def _resolve_schema(df, args, console):
    from mirrorbank.instruments.registry import REGISTRY, detect_instrument

    if args.instrument:
        if args.instrument not in REGISTRY:
            console.print(f"[red]Unknown instrument:[/red] {args.instrument!r}. Choose from: {', '.join(sorted(REGISTRY))}")
            sys.exit(1)
        return args.instrument, REGISTRY[args.instrument]()

    try:
        instrument_name = detect_instrument(df)
        console.print(f"[dim]Auto-detected instrument:[/dim] [cyan]{instrument_name}[/cyan]")
        return instrument_name, REGISTRY[instrument_name]()
    except ValueError as e:
        console.print(f"[red]Could not auto-detect instrument:[/red] {e}")
        sys.exit(1)


def cmd_generate(args: argparse.Namespace) -> None:
    from pathlib import Path

    import polars as pl
    from rich.console import Console

    from mirrorbank.generate.pipeline import generate, generate_to_csv
    from mirrorbank.privacy.budget import BudgetConfig

    console = Console()

    try:
        df = pl.read_csv(args.csv)
    except Exception as e:
        console.print(f"[red]Error reading CSV:[/red] {e}")
        sys.exit(1)

    instrument_name, schema = _resolve_schema(df, args, console)
    budget_config = BudgetConfig.from_preset(args.preset, dataset_size=df.height)

    out = args.out or f"outputs/{instrument_name}_synthetic.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if args.out and args.rows > 100_000:
        result = generate_to_csv(
            df, schema, n_rows=args.rows, budget_config=budget_config,
            seed=args.seed, model=args.model, path=out,
        )
        n_rows = args.rows
        epsilon_spent = result.epsilon_spent
        out_path = result.path
    else:
        result = generate(
            df, schema, n_rows=args.rows, budget_config=budget_config,
            seed=args.seed, model=args.model,
        )
        result.synthetic.write_csv(out)
        n_rows = result.n_rows
        epsilon_spent = result.epsilon_spent
        out_path = out

    console.print(f"[green]Generated[/green] {n_rows:,} synthetic rows for [cyan]{instrument_name}[/cyan]")
    console.print(f"  ε spent: [bold]{epsilon_spent:.3f}[/bold]")
    console.print(f"  Output: [bold]{out_path}[/bold]")


def cmd_evaluate(args: argparse.Namespace) -> None:
    import polars as pl
    from rich.console import Console

    from mirrorbank.evaluate.gauntlet import run_gauntlet

    console = Console()

    try:
        real = pl.read_csv(args.real)
        synth = pl.read_csv(args.synth)
    except Exception as e:
        console.print(f"[red]Error reading CSV:[/red] {e}")
        sys.exit(1)

    _instrument_name, schema = _resolve_schema(real, args, console)

    report = run_gauntlet(real, synth, schema=schema)
    report.print_summary()


def cmd_release(args: argparse.Namespace) -> None:
    import polars as pl
    from rich.console import Console

    from mirrorbank.evaluate.gauntlet import run_gauntlet
    from mirrorbank.generate.pipeline import generate
    from mirrorbank.privacy.budget import BudgetConfig

    console = Console()

    try:
        real = pl.read_csv(args.csv)
    except Exception as e:
        console.print(f"[red]Error reading CSV:[/red] {e}")
        sys.exit(1)

    instrument_name, schema = _resolve_schema(real, args, console)
    budget_config = BudgetConfig.from_preset(args.preset, dataset_size=real.height)

    result = generate(
        real, schema, n_rows=args.rows, budget_config=budget_config,
        seed=args.seed, model=args.model,
    )
    report = run_gauntlet(real, result.synthetic, schema=schema)

    from mirrorbank.release.bundler import bundle_release

    out_dir = args.out_dir or "outputs/release"
    bundle = bundle_release(
        result.synthetic,
        instrument=instrument_name,
        epsilon=result.epsilon_spent,
        delta=result.delta,
        out_dir=out_dir,
        report=report,
    )

    console.print(f"[green]Release bundle created[/green] for [cyan]{instrument_name}[/cyan] (version {bundle.version})")
    console.print(f"  Synthetic CSV:  [bold]{bundle.csv_path}[/bold]")
    console.print(f"  Certificate:    [bold]{bundle.certificate_path}[/bold]")
    console.print(f"  Scorecard:      [bold]{bundle.scorecard_path}[/bold]")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mirrorbank",
        description="Differentially private synthetic financial data generator",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # instruments
    p_inst = sub.add_parser("instruments", help="List supported payment instruments")
    p_inst.set_defaults(func=cmd_instruments)

    # profile
    p_prof = sub.add_parser("profile", help="Profile a dataset CSV")
    p_prof.add_argument("csv", help="Path to input CSV file")
    p_prof.add_argument("--instrument", "-i", help="Instrument name (auto-detected if omitted)")
    p_prof.set_defaults(func=cmd_profile)

    # budget
    p_bud = sub.add_parser("budget", help="Show privacy budget configuration")
    p_bud.add_argument("--preset", choices=["tight", "balanced", "loose", "demo"],
                       help="Show a specific preset (shows all if omitted)")
    p_bud.add_argument("--rows", type=int, default=100_000, help="Dataset size (default: 100,000)")
    p_bud.add_argument("--batch-size", type=int, default=4096, dest="batch_size",
                       help="Training batch size (default: 4096)")
    p_bud.set_defaults(func=cmd_budget)

    # generate
    p_gen = sub.add_parser("generate", help="Generate a synthetic dataset from a real CSV")
    p_gen.add_argument("csv", help="Path to input (real) CSV file")
    p_gen.add_argument("--instrument", "-i", help="Instrument name (auto-detected if omitted)")
    p_gen.add_argument("--preset", choices=["tight", "balanced", "loose", "demo"], default="balanced",
                       help="Privacy budget preset (default: balanced)")
    p_gen.add_argument("--rows", type=int, default=10_000, help="Number of synthetic rows to generate (default: 10,000)")
    p_gen.add_argument("--model", choices=["dp_statistical", "dp_ctgan"], default="dp_statistical",
                       help="Generation model (default: dp_statistical)")
    p_gen.add_argument("--out", help="Output CSV path (default: outputs/<instrument>_synthetic.csv)")
    p_gen.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p_gen.set_defaults(func=cmd_generate)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run the evaluation gauntlet on real vs synthetic data")
    p_eval.add_argument("--real", required=True, help="Path to real CSV file")
    p_eval.add_argument("--synth", required=True, help="Path to synthetic CSV file")
    p_eval.add_argument("--instrument", "-i", help="Instrument name (auto-detected from --real if omitted)")
    p_eval.set_defaults(func=cmd_evaluate)

    # release
    p_rel = sub.add_parser("release", help="Generate, evaluate, and bundle a synthetic data release")
    p_rel.add_argument("csv", help="Path to input (real) CSV file")
    p_rel.add_argument("--instrument", "-i", help="Instrument name (auto-detected if omitted)")
    p_rel.add_argument("--preset", choices=["tight", "balanced", "loose", "demo"], default="balanced",
                       help="Privacy budget preset (default: balanced)")
    p_rel.add_argument("--rows", type=int, default=10_000, help="Number of synthetic rows to generate (default: 10,000)")
    p_rel.add_argument("--model", choices=["dp_statistical", "dp_ctgan"], default="dp_statistical",
                       help="Generation model (default: dp_statistical)")
    p_rel.add_argument("--out-dir", dest="out_dir", help="Output directory for release bundle (default: outputs/release)")
    p_rel.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p_rel.set_defaults(func=cmd_release)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
