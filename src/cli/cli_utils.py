"""Shared CLI utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from playwright.sync_api import sync_playwright

from src.data_loading import UnifiedDataLoader
from src.data_loading.unified_data_loader import UnifiedDataLoadingError


def get_loader() -> UnifiedDataLoader:
    """Instantiate the data loader or fail gracefully with setup instructions."""
    try:
        return UnifiedDataLoader()
    except UnifiedDataLoadingError as e:
        typer.echo(f"\n[Error] {e}", err=True)
        typer.echo("-> Run 'awakenai unify-data' to generate the required dataset first.\n", err=True)
        raise typer.Exit(1)


def resolve_patients(
    patients: Optional[list[str]],
    all_patients: bool,
    loader: UnifiedDataLoader,
) -> list[str]:
    """Return final list of patient IDs based on CLI args."""
    if all_patients and patients:
        typer.echo("Error: provide either patient IDs or --all, not both.", err=True)
        raise typer.Exit(1)
    if all_patients:
        return loader.get_patient_ids()
    if not patients:
        typer.echo("Error: provide at least one patient ID or --all.", err=True)
        raise typer.Exit(1)
    return list(patients)


def print_table(df: pd.DataFrame, title: Optional[str] = None) -> None:
    """Print a DataFrame as a compact table to stdout."""
    if title:
        typer.echo(f"\n{title}")
        typer.echo("─" * len(title))
    typer.echo(df.to_string(index=False))
    typer.echo()


def generate_pdf_from_html(html_path: Path, pdf_path: Path) -> bool:
    """Uses Playwright to generate a PDF from an existing HTML file."""
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception:
                # If chromium is not installed, install it automatically
                typer.echo("  └─ PDF: First-time setup (downloading headless chromium)...")
                import subprocess

                subprocess.run(["python", "-m", "playwright", "install", "chromium"], check=True)
                browser = p.chromium.launch()

            page = browser.new_page()
            # Playwright needs an absolute file:// URI
            uri = html_path.absolute().as_uri()
            page.goto(uri, wait_until="networkidle")
            # Generate the PDF (A4 format, background enabled, etc to match standard browser print)
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "0.5in", "bottom": "0.5in", "left": "0.5in", "right": "0.5in"},
            )
            browser.close()
        return True
    except ImportError:
        typer.echo("  └─ PDF: Skipped (playwright not installed)", err=True)
        return False
    except Exception as e:
        typer.echo(f"  └─ PDF: Generation failed: {e}", err=True)
        return False


def print_report_paths(pid: str, sess: str, html_path: Optional[Path] = None, pdf_path: Optional[Path] = None) -> None:
    """Print the paths of generated HTML and PDF reports with tree formatting."""
    typer.echo(f"  {pid} / {sess}:")
    if html_path:
        typer.echo(f"    └─ HTML: {html_path}")
    if pdf_path:
        typer.echo(f"    └─ PDF:  {pdf_path}")
