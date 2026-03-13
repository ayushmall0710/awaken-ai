import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.reports.style_utils as style_utils
from src.data_loading import config
from src.pipelines.command_following import CONTRALATERAL_MAP, CommandFollowingAnalysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlotSpec:
    """Declarative description of a single per-side plot slot in the report."""

    key: str
    caption: str  # <figcaption> text shown above the image
    alt: str
    desc: str  # italic caption shown below the image
    error_msg: str


# ──────────────────────────────────────────────────────────────
#  Per-side plot specifications (order determines rendering order)
# ──────────────────────────────────────────────────────────────
_SIDE_PLOT_SPECS: List[PlotSpec] = [
    PlotSpec(
        key="psd",
        caption="PSD Overlay (Keep vs Stop)",
        alt="PSD Overlay",
        desc=(
            "Power spectrum comparison for the contralateral channel. "
            "Upper panel: absolute PSD. Lower panel: Stop &minus; Keep difference "
            "(green = ERD / desync, red = ERS / sync)."
        ),
        error_msg="PSD overlay unavailable. Not enough valid pairs for this command side.",
    ),
    # Not adding topomap for now: the pipeline loads only 3 ROI channels (C3, C4, Cz).
    # mne.viz.plot_topomap with 3 electrodes produces a meaningless triangular artefact.
    # Re-add when the pipeline is updated to load full-cap data.
]

# Columns shown in the contralateral-channel classification table (both CMD+ and CMD-)
_CONTRA_TABLE_COLS = ["side", "channel", "band", "erd_dB", "cohens_d", "p_value_raw", "significant"]
_CONTRA_TABLE_HEADERS = ["Side", "Channel", "Band", "ERD (dB)", "Cohen's d", "p (raw)", "Significant"]


class CommandFollowingReport:
    """Generates an HTML report for Command Following Analysis."""

    def __init__(
        self,
        cf_obj: CommandFollowingAnalysis,
        session_id: str,
        output_dir: Optional[Path] = None,
    ):
        self.cf_obj = cf_obj
        self.session_id = session_id

        if not hasattr(self.cf_obj, "patient_id") or not self.cf_obj.patient_id:
            logger.error("Pipeline has no patient_id. Cannot generate report.")
            raise ValueError("Pipeline patient_id is missing.")

        self._setup_output_dir(output_dir)

    def _setup_output_dir(self, output_dir: Optional[Path] = None) -> None:
        if output_dir is None:
            path_str = config.REPORT_DIR_TEMPLATE.format(
                patient_id=self.cf_obj.patient_id,
                session_id=self.session_id,
                pipeline_name="command_following",
                timestamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S"),
            )
            output_dir = Path(path_str)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report_file = self.output_dir / "report.html"

    # ──────────────────────────────────────────────────────────────
    #  CELL FORMATTERS
    # ──────────────────────────────────────────────────────────────

    def _format_cell(self, col: str, val: Any) -> str:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "N/A"
        if col in ("erd_dB", "erd_std", "cohens_d", "keep_mean_dB", "stop_mean_dB"):
            return f"{float(val):.3f}"
        if col in ("p_value_raw", "p_mixed", "p_value"):
            return "<0.001" if float(val) < 0.001 else f"{float(val):.3f}"
        if col == "accuracy":
            return f"{float(val) * 100:.1f}%"
        if col == "significant":
            return style_utils.ICON_TRUE if val else style_utils.ICON_FALSE
        return str(val)

    # ──────────────────────────────────────────────────────────────
    #  FULL ERD RESULTS TABLE
    # ──────────────────────────────────────────────────────────────

    _RESULTS_TABLE_SKIP = {"is_contralateral", "n_pairs"}

    def _build_results_table(self, erd_df: pd.DataFrame) -> str:
        if erd_df is None or erd_df.empty:
            return "<p>No ERD results calculated.</p>"

        headers = [c for c in erd_df.columns if c not in self._RESULTS_TABLE_SKIP]
        rows = [[self._format_cell(col, row[col]) for col in headers] for _, row in erd_df.iterrows()]
        return f"<div class='table-wrapper'>\n{style_utils.build_base_html_table(headers, rows)}\n</div>"

    # ──────────────────────────────────────────────────────────────
    #  RESULTS OVERVIEW
    # ──────────────────────────────────────────────────────────────

    def _build_results_overview(self, summary: dict, erd_df: Optional[pd.DataFrame]) -> str:
        """Top-of-report section: status cards + per-side ERD comparison table."""
        status = summary.get("cmd_status", "Unknown")
        n_pairs = summary.get("n_pairs", 0)
        left_pairs = summary.get("left_pairs", "?")
        right_pairs = summary.get("right_pairs", "?")
        chance = summary.get("classification_chance_level", 0.5) * 100

        badge_color = "#16a34a" if status == "CMD+" else "#dc2626"
        status_badge = style_utils.build_status_badge(status, bg_color=badge_color)

        cards = style_utils.build_metric_cards(
            [
                {
                    "title": "Classification",
                    "value": status_badge,
                    "desc": "CMD+ if significant contralateral desynchronization detected",
                },
                {
                    "title": "Pairs evaluated",
                    "value": str(n_pairs),
                    "desc": f"Left: {left_pairs} &nbsp;|&nbsp; Right: {right_pairs}",
                },
                {
                    "title": "Binomial chance",
                    "value": f"{chance:.1f}%",
                    "desc": f"Min accuracy to beat chance (N={n_pairs}, α=0.05)",
                },
            ]
        )

        # Per-side ERD comparison table
        comparison_html = ""
        if erd_df is not None and not erd_df.empty:
            comparison_html = "<h3 style='margin-top:1.5rem;'>ERD by Channel &amp; Band</h3>"
            for side in erd_df["side"].unique():
                side_df = erd_df[erd_df["side"] == side]
                contra_ch = CONTRALATERAL_MAP.get(side.lower(), "")
                comparison_html += (
                    f"<h4 style='margin-top:1rem;color:#4b2e83;'>"
                    f"{side.capitalize()} Command"
                    f"<small style='font-weight:normal;color:#64748b;margin-left:0.5rem;'>"
                    f"contralateral: {contra_ch}</small></h4>"
                )
                comparison_html += self._build_erd_comparison_table(side_df, contra_ch)

        return (
            f"<div class='metric-card' style='margin-bottom:2rem;'>"
            f"<h3 style='margin-top:0;'>Overall Test Results</h3>"
            f"{cards}"
            f"{comparison_html}"
            f"</div>"
        )

    def _build_erd_comparison_table(self, side_df: pd.DataFrame, contra_ch: str) -> str:
        """Compact table: Channel | Keep α | Stop α | α Diff | Keep β | Stop β | β Diff | Contra?"""
        bands = side_df["band"].unique()
        channels = side_df["channel"].unique()

        header_cells = ["<th>Channel</th>"]
        for band in bands:
            header_cells += [
                f"<th>Keep {band} (dB)</th>",
                f"<th>Stop {band} (dB)</th>",
                f"<th>{band} Diff</th>",
            ]

        rows_html = ""
        for ch in channels:
            row_cells = [f"<td><strong>{ch}</strong></td>"]
            for band in bands:
                row_df = side_df[(side_df["channel"] == ch) & (side_df["band"] == band)]
                if row_df.empty:
                    row_cells += ["<td>N/A</td>", "<td>N/A</td>", "<td>N/A</td>"]
                    continue

                r = row_df.iloc[0]
                keep_v = r.get("keep_mean_dB", float("nan"))
                stop_v = r.get("stop_mean_dB", float("nan"))
                diff_v = r.get("erd_dB", float("nan"))

                keep_str = f"{keep_v:.2f}" if np.isfinite(keep_v) else "N/A"
                stop_str = f"{stop_v:.2f}" if np.isfinite(stop_v) else "N/A"

                if np.isfinite(diff_v):
                    sign = "+" if diff_v > 0 else ""
                    diff_bg = "background:#dcfce7;color:#15803d;" if diff_v > 0 else "background:#fee2e2;color:#b91c1c;"
                    arrow = "↑ ERD" if diff_v > 0 else "↓ sync"
                    diff_cell = f"<td style='font-weight:bold;{diff_bg}'>{sign}{diff_v:.3f} {arrow}</td>"
                else:
                    diff_cell = "<td>N/A</td>"

                row_cells += [f"<td>{keep_str}</td>", f"<td>{stop_str}</td>", diff_cell]

            rows_html += f"<tr>{''.join(row_cells)}</tr>\n"

        return (
            "<div class='table-wrapper'><table>"
            f"<thead><tr>{''.join(header_cells)}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>"
        )

    # ──────────────────────────────────────────────────────────────
    #  LEGEND BOX
    # ──────────────────────────────────────────────────────────────

    def _build_legend_box(self) -> str:
        items = [
            {
                "term": "erd_dB",
                "desc": (
                    "Mean Event-Related Desynchronization (dB). Positive = desynchronization "
                    "during motor imagery (literature convention: ERD = Stop &minus; Keep)."
                ),
                "ranges": [
                    ("legend-excellent", "&gt; +2 dB &mdash; Strong"),
                    ("legend-good", "+1 to +2 dB &mdash; Good"),
                    ("legend-ok", "0 to +1 dB &mdash; Weak"),
                    ("legend-bad", "&lt; 0 dB &mdash; Synchronization (No motor activation)"),
                ],
            },
            {
                "term": "accuracy",
                "desc": "Single-trial classification accuracy for Keep vs Stop conditions.",
                "ranges": [
                    ("legend-excellent", "&ge; 70% &mdash; Excellent"),
                    ("legend-good", "60-70% &mdash; Good"),
                    ("legend-ok", "&gt; Chance &mdash; Acceptable"),
                    ("legend-bad", "&le; Chance &mdash; Random"),
                ],
            },
            {
                "term": "p_value_raw &amp; p_mixed",
                "desc": (
                    "<code>p_value_raw</code>: One-sided paired t-test (Stop &gt; Keep).<br/>"
                    "<code>p_mixed</code>: Random-intercept mixed model p-value."
                ),
                "ranges": [
                    ("legend-excellent", "&lt; 0.01 &mdash; Highly Significant"),
                    ("legend-good", "0.01-0.05 &mdash; Significant"),
                    ("legend-ok", "0.05-0.10 &mdash; Marginal"),
                    ("legend-bad", "&gt; 0.10 &mdash; Not Significant"),
                ],
            },
            {
                "term": "cohens_d",
                "desc": (
                    "Effect size of the paired difference. Positive indicates prominent motor imagery (Stop &gt; Keep)."
                ),
                "ranges": [
                    ("legend-excellent", "&gt; +0.8 &mdash; Large Effect"),
                    ("legend-good", "+0.5 to +0.8 &mdash; Medium Effect"),
                    ("legend-ok", "+0.2 to +0.5 &mdash; Small Effect"),
                    ("legend-bad", "&lt; +0.2 &mdash; Negligible"),
                ],
            },
            {
                "term": "channel",
                "desc": (
                    "Primary motor cortex (M1) mapped across central electrodes.<br/>"
                    "<strong>C3</strong>: Left hemisphere. <strong>C4</strong>: Right hemisphere. "
                    "<strong>Cz</strong>: Central midline."
                ),
            },
            {
                "term": "significant",
                "desc": (
                    f"{style_utils.ICON_TRUE} True / {style_utils.ICON_FALSE} False — "
                    "power difference between Keep and Stop is statistically significant "
                    "after Benjamini-Hochberg FDR correction (&alpha;=0.05). "
                    "Contralateral alignment is a separate criterion checked during CMD classification."
                ),
            },
        ]
        return style_utils.build_legend_box("Metrics Lexicon &amp; Interpretation", items)

    # ──────────────────────────────────────────────────────────────
    #  CSS EXTENSIONS
    # ──────────────────────────────────────────────────────────────

    def _build_css_extensions(self) -> str:
        return """
        .significant { color: #16a34a !important; font-weight: bold; }

        /* One card per plot type; figures inside sit side by side */
        .plot-card {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-top: 4px solid #4b2e83;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .plot-card h3 { margin-top: 0; }
        .plot-card .side-figures {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }
        .plot-card .side-figures figure {
            flex: 1 1 0;
            min-width: 300px;
            margin: 0;
        }
        .plot-card figcaption {
            font-weight: bold; text-align: center;
            color: #4b2e83; margin-bottom: 0.5rem;
        }
        .plot-card img {
            max-width: 100%; height: auto;
            border: 1px solid #e2e8f0; border-radius: 4px;
            display: block; margin: 0 auto;
        }
        .plot-desc {
            font-size: 0.85rem; color: #555;
            text-align: center; margin-top: 0.4rem; font-style: italic;
        }
        .plot-error {
            background: #fef2f2;
            border: 1px solid #fca5a5;
            border-radius: 6px;
            padding: 0.75rem 1.25rem;
            color: #991b1b;
            font-size: 0.85rem;
        }
        """

    # ──────────────────────────────────────────────────────────────
    #  CLASSIFICATION CARD
    # ──────────────────────────────────────────────────────────────

    def _render_contra_table(self, df: pd.DataFrame, desc: str) -> str:
        """Shared table renderer for contralateral-channel classification results.

        Used for both CMD+ (all rows are significant) and CMD- (sorted by p-value).
        Shows ✓/✗ icons in the `significant` column instead of True/False.
        """
        rows = [[self._format_cell(c, row[c]) for c in _CONTRA_TABLE_COLS] for _, row in df.iterrows()]
        return (
            f"<p style='color:#64748b;margin-bottom:0.5rem;font-size:0.9rem;'>{desc}</p>"
            + style_utils.build_base_html_table(_CONTRA_TABLE_HEADERS, rows)
        )

    def _build_classification_results(self, summary: dict) -> str:
        """CMD+/- decision card.

        CMD+: shows the channels that passed all four criteria.
        CMD-: shows the best contralateral candidates sorted by raw p-value so
        the clinician can see how close (or far) the patient was.
        Both cases use the same table columns — only the description text differs.
        """
        status = summary.get("cmd_status", "Unknown")
        sig_results = summary.get("significant_results", [])
        badge_color = "#16a34a" if status == "CMD+" else "#dc2626"
        badge = style_utils.build_status_badge(status, bg_color=badge_color)

        df = self.cf_obj.erd_results
        if status == "CMD+" and sig_results:
            sig_df = pd.DataFrame(sig_results).sort_values("p_value_raw")
            body = self._render_contra_table(
                sig_df,
                "Channels meeting all four criteria "
                "(contralateral, FDR-significant, ERD &gt; +1 dB, Cohen's d &gt; 0.5):",
            )
        elif df is not None and not df.empty:
            contra_df = df[df["is_contralateral"]].sort_values("p_value_raw")
            if not contra_df.empty:
                body = self._render_contra_table(
                    contra_df,
                    "No channel met all four classification criteria. "
                    "Contralateral channels below, sorted by raw p-value:",
                )
            else:
                body = "<p style='color:#64748b;margin-top:0.75rem;'>No contralateral results available.</p>"
        else:
            body = "<p style='color:#64748b;margin-top:0.75rem;'>No ERD results computed.</p>"

        return (
            f"<div class='metric-card' style='margin-bottom:2rem;'>"
            f"<h3 style='margin-top:0;'>CMD Classification &nbsp; {badge}</h3>"
            f"{body}"
            f"</div>"
        )

    # ──────────────────────────────────────────────────────────────
    #  PLOTS SECTION (HTML rendering only — no file I/O here)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _render_figure(path: Optional[str], caption: str, alt: str, error_msg: str) -> str:
        """Render one <figure> — image with figcaption, or an error box.

        desc is intentionally excluded: when multiple figures share a card, the
        description belongs to the card (shown once, centered below all figures)
        rather than duplicated inside every figure.
        """
        if path:
            return f"<figure><figcaption>{caption}</figcaption><img src='{path}' alt='{alt}'/></figure>"
        return f"<figure><div class='plot-error'><strong>{error_msg}</strong></div></figure>"

    def _build_plots_section(self, img_paths: dict) -> str:
        """Render all visualizations to HTML.

        One card per plot type. Left and right command images sit side by side
        inside each card. The shared description renders once, centered below
        both images.
        """
        plots_html = ""
        sides = sorted(img_paths.get("psd", {}).keys())

        # Bar plot — overall across sides
        bar_path = img_paths.get("bar")
        if bar_path:
            plots_html += (
                f"<div class='plot-card'>"
                f"<h3>Average ERD Bar Plot</h3>"
                f"<figure style='margin:0;'>"
                f"<img style='max-width:100%;' src='{bar_path}' alt='ERD Bar Plot'/>"
                f"</figure>"
                f"<div class='plot-desc'>Average ERD (dB) per channel and band. "
                f"Positive = desynchronization during motor imagery. "
                f"Error bars show &plusmn;1 SD.</div>"
                f"</div>"
            )

        if not sides:
            return plots_html

        # One card per plot type; left | right figures, shared desc below
        for spec in _SIDE_PLOT_SPECS:
            figures_html = "".join(
                self._render_figure(
                    path=img_paths.get(spec.key, {}).get(side),
                    caption=f"{side.capitalize()} Command",
                    alt=f"{spec.alt} — {side.capitalize()}",
                    error_msg=spec.error_msg,
                )
                for side in sides
            )
            plots_html += (
                f"<div class='plot-card'>"
                f"<h3>{spec.caption}</h3>"
                f"<div class='side-figures'>{figures_html}</div>"
                f"<div class='plot-desc'>{spec.desc}</div>"
                f"</div>"
            )

        return plots_html

    # ──────────────────────────────────────────────────────────────
    #  PLOT GENERATION (file I/O — split into focused sub-functions)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _save_fig(fig: plt.Figure, path: Path) -> str:
        """Save and close a matplotlib figure; return cross-platform relative POSIX path."""
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def _generate_bar_plot(self, viz_dir: Path, erd_df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Save the group-level ERD bar chart. Returns absolute file:// URI."""
        if erd_df is None:
            return None
        fig = self.cf_obj.viz.plot_erd_bar(erd_df, CONTRALATERAL_MAP)
        path = self._save_fig(fig, viz_dir / "erd_bar.png")
        return path.resolve().as_uri()

    def _generate_side_plots(self, viz_dir: Path, side: str, keep_epochs, stop_epochs) -> dict:
        """Save PSD overlay for one command side. Returns {"psd": absolute file:// URI}."""
        contra_ch = CONTRALATERAL_MAP[side]
        side_label = side.capitalize()

        fig_psd = self.cf_obj.viz.plot_psd_overlay(
            keep_epochs,
            stop_epochs,
            channel=contra_ch,
            title=f"PSD Overlay (Keep vs Stop) — {side_label} Command (Ch: {contra_ch})",
        )
        path = self._save_fig(fig_psd, viz_dir / f"psd_overlay_{side}.png")
        return {"psd": path.resolve().as_uri()}

    def _generate_plots(self, erd_df: pd.DataFrame) -> dict:
        """Orchestrate plot generation across all sides."""
        viz_dir = self.output_dir / "plots"
        viz_dir.mkdir(exist_ok=True)

        img_paths: dict = {
            "bar": self._generate_bar_plot(viz_dir, erd_df),
            "psd": {},
        }

        for side in sorted({p.side for p in self.cf_obj.pairs}):
            keep_epochs, stop_epochs = self.cf_obj.get_stacked_epochs(side)
            side_paths = self._generate_side_plots(viz_dir, side, keep_epochs, stop_epochs)
            for key, path in side_paths.items():
                img_paths[key][side] = path

        return img_paths

    # ──────────────────────────────────────────────────────────────
    #  MAIN GENERATE METHOD
    # ──────────────────────────────────────────────────────────────

    def _run_and_collect(self) -> tuple:
        """Run plots generation and collect all data needed to build HTML.

        Centralises the three calls shared by generate() and build_session_html()
        so neither method duplicates the logic.
        """
        summary = self.cf_obj.generate_summary()
        erd_df = self.cf_obj.erd_results
        img_paths = self._generate_plots(erd_df)
        return summary, erd_df, img_paths

    def _build_content_html(self, summary: dict, erd_df, img_paths: dict) -> str:
        """Return the bare body content (tables, legend, plots) with no HTML doc wrapper."""
        return f"""
            {self._build_results_overview(summary, erd_df)}

            {self._build_classification_results(summary)}

            <h2 style="margin-bottom:0">Full ERD Results</h2>
            {self._build_results_table(erd_df)}

            {self._build_legend_box()}

            <h2 style="margin-bottom:0">Motor Imagery Visualizations</h2>
            {self._build_plots_section(img_paths)}
        """

    def build_session_html(self) -> str:
        """Return a self-contained session fragment: collapsible panel + content.

        Rendered as a native HTML ``<details open>`` so the session starts
        expanded. Clicking the session panel header (the ``<summary>``) toggles
        the content. The ▼ arrow rotates to ▲ via CSS transition — no JS needed.

        This is the fragment consumed by style_utils.stitch_and_save() in the
        combined multi-session report flow.
        """
        summary, erd_df, img_paths = self._run_and_collect()
        return style_utils.wrap_session_fragment(self.session_id, self._build_content_html(summary, erd_df, img_paths))

    def generate(self) -> Path:
        """Write a standalone single-session HTML report to disk."""
        summary, erd_df, img_paths = self._run_and_collect()

        html = style_utils.build_html_header(
            title="Command Following Analysis Report",
            patient_id=self.cf_obj.patient_id,
            session_id=self.session_id,
            extra_css=self._build_css_extensions(),
        )
        html += self._build_content_html(summary, erd_df, img_paths)
        html += style_utils.build_html_footer("Command Following Pipeline")

        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("Command Following report generated at %s", self.report_file)
        return self.report_file
