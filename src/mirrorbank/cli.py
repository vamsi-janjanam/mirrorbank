"""
Mirrorbank CLI — entry point for running the pipeline from the command line.

Usage:
    mirrorbank instruments                          # list supported payment instruments
    mirrorbank profile <csv> [--instrument NAME]    # profile a dataset
    mirrorbank budget --preset balanced --rows 100000  # show privacy budget config
"""

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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
