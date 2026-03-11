"""HTML Report for the Claassen SVM Command Following pipeline."""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np

import src.reports.style_utils as style_utils
from src.data_loading import config
from src.pipelines.command_following_claassen import CommandFollowingClaassen
from src.viz.command_following_claassen_viz import ClaassenVisualizer

logger = logging.getLogger(__name__)

_RESULTS_TABLE_COLS = [
    "side",
    "n_pairs",
    "n_channels",
    "n_features",
    "auc",
    "accuracy",
    "chance_level",
    "p_value_perm",
    "significant",
]
_RESULTS_TABLE_HEADERS = [
    "Side",
    "Pairs",
    "Channels",
    "Features",
    "AUC",
    "Accuracy",
    "Chance",
    "p (perm)",
    "Significant",
]


class CommandFollowingClaassenReport:
    """Generates an HTML report for the Claassen SVM Command Following pipeline."""

    def __init__(
        self,
        pipeline: CommandFollowingClaassen,
        session_id: str,
        output_dir: Optional[Path] = None,
    ):
        self.pipeline = pipeline
        self.session_id = session_id

        if not hasattr(self.pipeline, "patient_id") or not self.pipeline.patient_id:
            raise ValueError("Pipeline patient_id is missing.")

        self._setup_output_dir(output_dir)
        self.viz = ClaassenVisualizer(pipeline.bands)

    def _setup_output_dir(self, output_dir: Optional[Path] = None) -> None:
        if output_dir is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            path_str = config.REPORT_DIR_TEMPLATE.format(
                patient_id=self.pipeline.patient_id,
                session_id=self.session_id,
                pipeline_name="command_following_svm",
                timestamp=timestamp,
            )
            output_dir = Path(path_str)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_file = self.output_dir / "report.html"

    # ──────────────────────────────────────────────────────────────
    #  Cell Formatting
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _format_cell(col: str, val: Any) -> str:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "N/A"
        if col == "auc":
            return f"{float(val):.3f}"
        if col in ("accuracy", "chance_level"):
            return f"{float(val) * 100:.1f}%"
        if col == "p_value_perm":
            return "<0.001" if float(val) < 0.001 else f"{float(val):.4f}"
        if col == "significant":
            return style_utils.ICON_TRUE if val else style_utils.ICON_FALSE
        return str(val)

    # ──────────────────────────────────────────────────────────────
    #  Results Overview Cards
    # ──────────────────────────────────────────────────────────────

    def _build_results_overview(self, summary: Dict[str, Any]) -> str:
        status = summary.get("cmd_status", "Unknown")
        n_pairs = summary.get("n_pairs", 0)
        left_pairs = summary.get("left_pairs", "?")
        right_pairs = summary.get("right_pairs", "?")
        n_perms = summary.get("n_permutations", "?")

        badge_color = "#16a34a" if status == "CMD+" else "#dc2626"
        status_badge = style_utils.build_status_badge(status, bg_color=badge_color)

        return style_utils.build_metric_cards(
            [
                {
                    "title": "Classification",
                    "value": status_badge,
                    "desc": "CMD+ if any side has permutation p < 0.05",
                },
                {
                    "title": "Pairs Evaluated",
                    "value": str(n_pairs),
                    "desc": f"Left: {left_pairs} &nbsp;|&nbsp; Right: {right_pairs}",
                },
                {
                    "title": "Permutations",
                    "value": str(n_perms),
                    "desc": "Label shuffles for null distribution",
                },
            ]
        )

    # ──────────────────────────────────────────────────────────────
    #  Per-side Results Table
    # ──────────────────────────────────────────────────────────────

    def _build_results_table(self, details_df) -> str:
        if details_df is None or details_df.empty:
            return "<p>No SVM results calculated.</p>"

        rows = [[self._format_cell(col, row[col]) for col in _RESULTS_TABLE_COLS] for _, row in details_df.iterrows()]
        return f"<div class='table-wrapper'>\n{style_utils.build_base_html_table(_RESULTS_TABLE_HEADERS, rows)}\n</div>"

    # ──────────────────────────────────────────────────────────────
    #  Per-side Detail Cards
    # ──────────────────────────────────────────────────────────────

    def _build_side_cards(self, summary: Dict[str, Any]) -> str:
        side_results = summary.get("side_results", [])
        if not side_results:
            return ""

        cards_data = []
        for sr in side_results:
            sig = sr.get("significant", False)
            border_color = "#16a34a" if sig else "#dc2626"  # Green or Red
            icon = "✓" if sig else "✗"
            label = "Significant" if sig else "Not significant"

            cards_data.append(
                {
                    "title": f"{sr['side'].capitalize()} Command",
                    "value": f"AUC = {sr['auc']:.3f}",
                    "desc": (
                        f"Accuracy: {sr['accuracy']:.1%} &nbsp;|&nbsp; "
                        f"p = {sr['p_value_perm']:.4f} &nbsp;|&nbsp; "
                        f"{icon} {label}"
                    ),
                    "border_color": border_color,
                }
            )

        html = "<h3>Per-Side Classification</h3>\n"
        html += style_utils.build_metric_cards(cards_data)
        return html

    # ──────────────────────────────────────────────────────────────
    #  Legend Box
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_legend_box() -> str:
        items = [
            {
                "term": "AUC (ROC)",
                "desc": (
                    "Area Under the ROC Curve from Leave-One-Out cross-validated SVM. "
                    "Measures how well the classifier separates keep (motor imagery) from stop (rest)."
                ),
                "ranges": [
                    ("legend-excellent", "&gt; 0.80 &mdash; Strong"),
                    ("legend-good", "0.70&ndash;0.80 &mdash; Good"),
                    ("legend-ok", "0.60&ndash;0.70 &mdash; Weak"),
                    ("legend-bad", "&le; 0.50 &mdash; Chance"),
                ],
            },
            {
                "term": "p (perm)",
                "desc": (
                    "Permutation p-value: fraction of label-shuffled AUCs that exceed "
                    "the observed AUC. Accounts for small-sample bias."
                ),
                "ranges": [
                    ("legend-excellent", "&lt; 0.01 &mdash; Highly Significant"),
                    ("legend-good", "0.01&ndash;0.05 &mdash; Significant"),
                    ("legend-ok", "0.05&ndash;0.10 &mdash; Marginal"),
                    ("legend-bad", "&gt; 0.10 &mdash; Not Significant"),
                ],
            },
            {
                "term": "Accuracy",
                "desc": "LOO-CV classification accuracy (keep vs stop).",
                "ranges": [
                    ("legend-excellent", "&ge; 70% &mdash; Excellent"),
                    ("legend-good", "60&ndash;70% &mdash; Good"),
                    ("legend-ok", "&gt; Chance &mdash; Acceptable"),
                    ("legend-bad", "&le; Chance &mdash; Random"),
                ],
            },
            {
                "term": "CMD+/CMD&minus;",
                "desc": (
                    "<strong>CMD+</strong>: At least one command side has permutation p &lt; 0.05. "
                    "<strong>CMD&minus;</strong>: No side reaches significance. "
                    "Uses all available EEG channels (CLINICAL_20) as features."
                ),
            },
        ]
        return style_utils.build_legend_box("Metrics Lexicon &amp; Interpretation", items)

    # ──────────────────────────────────────────────────────────────
    #  CSS Extensions (plot cards)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_css_extensions() -> str:
        return """
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
        .plot-card img {
            max-width: 100%; height: auto;
            border: 1px solid #e2e8f0; border-radius: 4px;
            display: block; margin: 0 auto;
        }
        .plot-desc {
            font-size: 0.85rem; color: #555;
            text-align: center; margin-top: 0.4rem; font-style: italic;
        }
        """

    # ──────────────────────────────────────────────────────────────
    #  Plot Generation
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _save_fig(fig: plt.Figure, path: Path) -> str:
        """Save and close a matplotlib figure; return file:// URI."""
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return path.resolve().as_uri()

    def _generate_plots(self, side_results: List[Dict]) -> Dict[str, Optional[str]]:
        """Generate all plots and return a dict of plot_name -> file:// URI."""
        viz_dir = self.output_dir / "plots"
        viz_dir.mkdir(exist_ok=True)
        img_paths: Dict[str, Optional[str]] = {}

        if not side_results:
            return img_paths

        # ROC Curves (combined figure with per-side subplots)
        fig_roc = self.viz.plot_roc_curves(side_results)
        img_paths["roc"] = self._save_fig(fig_roc, viz_dir / "roc_curves.png")

        # Permutation null distribution (combined figure with per-side subplots)
        fig_perm = self.viz.plot_permutation_distributions(side_results)
        img_paths["perm"] = self._save_fig(fig_perm, viz_dir / "permutation_distributions.png")

        # Channel weight topomap — needs MNE Info with valid electrode positions
        epochs_info = self._get_epochs_info(side_results)
        if epochs_info is not None:
            try:
                fig_topo = self.viz.plot_channel_weight_topomaps(side_results, epochs_info)
                img_paths["topo"] = self._save_fig(fig_topo, viz_dir / "channel_weights_topomap.png")
            except Exception as e:
                logger.warning("Channel weight topomap failed: %s — skipping.", e)
        else:
            logger.warning("No epochs info available — skipping channel weight topomap.")

        return img_paths

    @staticmethod
    def _get_epochs_info(side_results: List[Dict]) -> Optional[mne.Info]:
        """Build an MNE Info with standard 10-20 montage for topomap rendering."""
        if not side_results:
            return None
        channels = side_results[0].get("channels_used", [])
        if not channels:
            return None
        info = mne.create_info(ch_names=channels, sfreq=256, ch_types="eeg")
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, match_case=False, on_missing="warn")
        return info

    def _build_plots_section(self, img_paths: Dict[str, Optional[str]]) -> str:
        """Render the plots section of the report."""
        if not img_paths:
            return ""

        html = "<h2 style='margin-bottom:0'>SVM Classification Visualizations</h2>\n"

        # ROC Curves
        if img_paths.get("roc"):
            html += (
                "<div class='plot-card'>"
                "<h3>ROC Curves</h3>"
                f"<img src='{img_paths['roc']}' alt='ROC Curves'/>"
                "<div class='plot-desc'>"
                "Receiver Operating Characteristic curves from Leave-One-Out cross-validated SVM. "
                "Higher AUC indicates better separation between keep (motor imagery) and stop (rest)."
                "</div></div>"
            )

        # Permutation distribution
        if img_paths.get("perm"):
            html += (
                "<div class='plot-card'>"
                "<h3>Permutation Null Distribution</h3>"
                f"<img src='{img_paths['perm']}' alt='Permutation Null Distribution'/>"
                "<div class='plot-desc'>"
                "Histogram of AUCs from label-shuffled data (null distribution). "
                "The red dashed line shows the observed AUC. "
                "If the observed AUC falls far to the right, the classification is unlikely due to chance."
                "</div></div>"
            )

        # Channel weight topomap
        if img_paths.get("topo"):
            html += (
                "<div class='plot-card'>"
                "<h3>SVM Channel Importance</h3>"
                f"<img src='{img_paths['topo']}' alt='SVM Channel Importance Topomap'/>"
                "<div class='plot-desc'>"
                "Mean absolute SVM weight per channel, averaged across frequency bands. "
                "Warmer colors indicate channels that contributed more to the classification. "
                "Motor cortex channels (C3, C4) should show higher importance for genuine command following."
                "</div></div>"
            )

        return html

    # ──────────────────────────────────────────────────────────────
    #  Content + Document Assembly
    # ──────────────────────────────────────────────────────────────

    def _collect_side_results(self) -> List[Dict]:
        """Extract the full side results (with y_true, y_scores, etc.) from the pipeline."""
        if not self.pipeline.svm_results:
            return []
        return self.pipeline.svm_results.get("side_results", [])

    def _build_content_html(self, summary: Dict[str, Any], details_df, img_paths: Dict) -> str:
        return (
            f"<div class='metric-card' style='margin-bottom:2rem;'>"
            f"<h3 style='margin-top:0;'>Overall Test Results</h3>"
            f"{self._build_results_overview(summary)}"
            f"</div>"
            f"{self._build_side_cards(summary)}"
            f"<h3>Detailed Results</h3>"
            f"{self._build_results_table(details_df)}"
            f"{self._build_legend_box()}"
            f"{self._build_plots_section(img_paths)}"
        )

    def build_session_html(self) -> str:
        """Return a collapsible session fragment for combined reports."""
        summary = self.pipeline.generate_summary()
        details_df = self.pipeline.svm_results.get("details") if self.pipeline.svm_results else None
        side_results = self._collect_side_results()
        img_paths = self._generate_plots(side_results)
        panel = style_utils.build_session_panel(self.session_id, collapsible=True)
        content = self._build_content_html(summary, details_df, img_paths)
        return f"<details class='session-wrapper' open>{panel}<div class='session-content'>{content}</div></details>"

    def generate(self) -> Path:
        """Write a standalone single-session HTML report to disk."""
        summary = self.pipeline.generate_summary()
        details_df = self.pipeline.svm_results.get("details") if self.pipeline.svm_results else None
        side_results = self._collect_side_results()
        img_paths = self._generate_plots(side_results)

        html = style_utils.build_html_header(
            title="Command Following (SVM) Analysis Report",
            patient_id=self.pipeline.patient_id,
            session_id=self.session_id,
            extra_css=self._build_css_extensions(),
        )
        html += self._build_content_html(summary, details_df, img_paths)
        html += style_utils.build_html_footer("Command Following SVM Pipeline")

        with open(self.report_file, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("SVM Command Following report generated at %s", self.report_file)
        return self.report_file
