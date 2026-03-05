"""HTML report for Language Tracking Analysis."""

import base64
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

import src.reports.style_utils as style_utils
from src.data_loading import config
from src.pipelines.language_tracking import LanguageTrackingAnalysis
from src.viz.language_plots import plot_itpc_results, plot_itpc_spectrum, plot_itpc_topomap

logger = logging.getLogger(__name__)

_ENTRAINMENT_COLS = [
    ("Patient", "patient_id"),
    ("Trials", "n_trials"),
    ("ITPC Word (3.125 Hz)", "itpc_word"),
    ("ITPC Phrase (1.56 Hz)", "itpc_phrase"),
    ("ITPC Sentence (0.78 Hz)", "itpc_sentence"),
    ("Comprehension Combined", "itpc_comprehension_combined"),
    ("Cognitive/Acoustic Ratio", "ratio_cognitive_acoustic"),
    ("p Word", "dft_p_word"),
    ("p Phrase", "dft_p_phrase"),
    ("p Sentence", "dft_p_sentence"),
    ("p Comprehension", "dft_p_comprehension"),
]


_TARGET_FREQS = [
    (LanguageTrackingAnalysis.TARGET_WORD_FREQ, "Word"),
    (LanguageTrackingAnalysis.TARGET_PHRASE_FREQ, "Phrase"),
    (LanguageTrackingAnalysis.TARGET_SENTENCE_FREQ, "Sentence"),
]

_MORLET_METRICS = {
    "freq_sentence_hz": LanguageTrackingAnalysis.TARGET_SENTENCE_FREQ,
    "freq_phrase_hz": LanguageTrackingAnalysis.TARGET_PHRASE_FREQ,
    "freq_word_hz": LanguageTrackingAnalysis.TARGET_WORD_FREQ,
}

_PLOT_CSS = """
        .plot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .plot-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 0.75rem;
            text-align: center;
        }
        .plot-card figcaption {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.4rem;
        }
        """


class LanguageTrackingReport:
    """Generates an HTML report for Language Tracking Analysis."""

    def __init__(self, lt_obj, session_id: str, output_dir: Optional[Path] = None):
        self.lt_obj = lt_obj
        self.session_id = session_id

        if not getattr(self.lt_obj, "patient_id", None):
            raise ValueError("Pipeline patient_id is missing.")

        if output_dir is None:
            output_dir = config.REPORTS_DIR / self.lt_obj.patient_id / self.session_id / "language_tracking"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_file = self.output_dir / f"{self.session_id}_language_report.html"
        self._plot_paths: Optional[dict] = None

    def _save_plots(self) -> dict:
        """Generate all plots to output_dir. Returns {key: Path}. Result is cached."""
        if self._plot_paths is not None:
            return self._plot_paths

        row = self.lt_obj.results.iloc[0]
        pid = self.lt_obj.patient_id
        spectrum = self.lt_obj._dft_spectrum_full
        freqs = self.lt_obj._dft_freqs
        info = self.lt_obj._dft_info
        metrics = row.to_dict()
        paths = {}

        paths["itpc_spectrum"] = plot_itpc_spectrum(spectrum, freqs, pid, self.output_dir, metrics)

        word_idx = int(np.argmin(np.abs(freqs - LanguageTrackingAnalysis.TARGET_WORD_FREQ)))
        vmax = max(float(np.percentile(spectrum[:, word_idx], 95)) * 1.2, 0.1)
        vlim = (0.0, vmax)

        for freq, label in _TARGET_FREQS:
            paths[f"topomap_{label.lower()}"] = plot_itpc_topomap(
                spectrum, freqs, info, freq, label, pid, self.output_dir, vlim=vlim
            )

        if self.lt_obj._morlet_itc is not None:
            try:
                plot_itpc_results(self.lt_obj._morlet_itc, pid, str(self.output_dir), _MORLET_METRICS)
                tfr_path = self.output_dir / f"{pid}_language_ITPC_tfr.png"
                if tfr_path.exists():
                    paths["morlet_tfr"] = tfr_path

                morlet_data = self.lt_obj._morlet_itc.data  # (n_channels, n_freqs, n_times)
                morlet_spectrum = np.mean(morlet_data, axis=-1)  # (n_channels, n_freqs)
                morlet_freqs = self.lt_obj._morlet_itc.freqs

                morlet_metrics = {
                    "itpc_word": float(row.get("morlet_itpc_word", 0)),
                    "itpc_phrase": float(row.get("morlet_itpc_phrase", 0)),
                    "itpc_sentence": float(row.get("morlet_itpc_sentence", 0)),
                    "p_word": float(row.get("morlet_p_word", 1)),
                    "p_phrase": float(row.get("morlet_p_phrase", 1)),
                    "p_sentence": float(row.get("morlet_p_sentence", 1)),
                }
                paths["morlet_spectrum"] = plot_itpc_spectrum(
                    morlet_spectrum, morlet_freqs, pid, self.output_dir, morlet_metrics, method_label="Morlet"
                )

                morlet_word_idx = int(np.argmin(np.abs(morlet_freqs - LanguageTrackingAnalysis.TARGET_WORD_FREQ)))
                morlet_vmax = max(float(np.percentile(morlet_spectrum[:, morlet_word_idx], 95)) * 1.2, 0.1)
                morlet_vlim = (0.0, morlet_vmax)

                for freq, label in _TARGET_FREQS:
                    paths[f"morlet_topomap_{label.lower()}"] = plot_itpc_topomap(
                        morlet_spectrum,
                        morlet_freqs,
                        info,
                        freq,
                        label,
                        pid,
                        self.output_dir,
                        vlim=morlet_vlim,
                        method_label="Morlet",
                    )
            except Exception as e:
                logger.warning(f"Morlet plots failed: {e}")

        self._plot_paths = paths
        return paths

    @staticmethod
    def _embed_image(path: Path, alt: str = "") -> str:
        try:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"<img src='data:image/png;base64,{b64}' alt='{alt}' style='width:100%;max-width:100%;' />"
        except (FileNotFoundError, OSError):
            return f"<p><em>Plot unavailable: {alt}</em></p>"

    def _format_cell(self, key: str, val: Any) -> str:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "N/A"
        if key in (
            "itpc_word",
            "itpc_phrase",
            "itpc_sentence",
            "itpc_comprehension_combined",
            "lh_itpc_word",
            "lh_itpc_phrase",
            "lh_itpc_sentence",
            "rh_itpc_word",
            "rh_itpc_phrase",
            "rh_itpc_sentence",
            "ratio_cognitive_acoustic",
        ):
            return f"{float(val):.4f}"
        if key.startswith("dft_p_"):
            return "<0.001" if float(val) < 0.001 else f"{float(val):.3f}"
        if key.startswith("lateralization_index_"):
            v = float(val)
            color = "#16a34a" if v > 0.1 else ("#dc2626" if v < -0.1 else "#856404")
            return f"<span style='color:{color};font-weight:bold;'>{v:+.3f}</span>"
        return str(val)

    def _build_css_extensions(self) -> str:
        return _PLOT_CSS

    def _build_overview_cards(self) -> str:
        row = self.lt_obj.results.iloc[0]
        n_trials = int(row.get("n_trials", 0))
        itpc_comp = float(row.get("itpc_comprehension_combined", 0))
        p_comp = float(row.get("dft_p_comprehension", 1.0))
        li_comp = float(row.get("lateralization_index_comprehension", 0.0))

        badge = (
            style_utils.build_status_badge("Significant", bg_color="#16a34a")
            if p_comp < 0.05
            else style_utils.build_status_badge("Not Significant", bg_color="#dc2626")
        )

        if li_comp > 0.1:
            li_badge = style_utils.build_status_badge(f"LI={li_comp:+.3f} Left", bg_color="#2563eb")
        elif li_comp < -0.1:
            li_badge = style_utils.build_status_badge(f"LI={li_comp:+.3f} Right", bg_color="#7c3aed")
        else:
            li_badge = style_utils.build_status_badge(f"LI={li_comp:+.3f} Bilateral", bg_color="#856404")

        return style_utils.build_metric_cards(
            [
                {
                    "title": "Trials Analyzed",
                    "value": str(n_trials),
                    "desc": "Language epochs after artifact rejection",
                },
                {
                    "title": "Comprehension ITPC",
                    "value": f"{itpc_comp:.4f}",
                    "desc": f"Combined phrase+sentence rate (p={p_comp:.3f})",
                },
                {
                    "title": "Comprehension Tracking",
                    "value": badge,
                    "desc": "Permutation test at phrase+sentence rates",
                },
                {
                    "title": "Lateralization",
                    "value": li_badge,
                    "desc": "No handedness assumed — interpret with clinical context",
                },
            ]
        )

    def _build_entrainment_table(self) -> str:
        row = self.lt_obj.results.iloc[0]
        headers = [name for name, _ in _ENTRAINMENT_COLS]
        cells = [self._format_cell(key, row.get(key)) for _, key in _ENTRAINMENT_COLS]
        table = style_utils.build_base_html_table(headers, [cells])
        return f"<h3>Global Entrainment Metrics</h3><div class='table-wrapper'>{table}</div>"

    def _build_lateralization_section(self) -> str:
        row = self.lt_obj.results.iloc[0]
        rates = [
            ("Word (3.125 Hz)", "lh_itpc_word", "rh_itpc_word", "lateralization_index_word"),
            ("Phrase (1.56 Hz)", "lh_itpc_phrase", "rh_itpc_phrase", "lateralization_index_phrase"),
            ("Sentence (0.78 Hz)", "lh_itpc_sentence", "rh_itpc_sentence", "lateralization_index_sentence"),
            ("Comprehension", None, None, "lateralization_index_comprehension"),
        ]
        headers = [
            "Rate",
            "LH ITPC (Fp1/F7/T7/F3/C3/P3)",
            "RH ITPC (Fp2/F8/T8/F4/C4/P4)",
            "Lateralization Index",
        ]
        table_rows = []
        for label, lh_key, rh_key, li_key in rates:
            lh_val = self._format_cell(lh_key, row.get(lh_key)) if lh_key else "&mdash;"
            rh_val = self._format_cell(rh_key, row.get(rh_key)) if rh_key else "&mdash;"
            li_val = self._format_cell(li_key, row.get(li_key))
            table_rows.append([label, lh_val, rh_val, li_val])

        table = style_utils.build_base_html_table(headers, table_rows)
        note = (
            "<p style='font-size:0.8rem;color:#64748b;margin-top:0.5rem;'>"
            "LI = (LH &minus; RH) / (LH + RH). Positive = left-dominant. "
            "No handedness assumption is made &mdash; interpret alongside clinical context."
            "</p>"
        )
        return f"<h3>Hemisphere Lateralization Analysis</h3><div class='table-wrapper'>{table}</div>{note}"

    def _build_plots_section(self, plot_paths: dict) -> str:
        sections = []

        if "itpc_spectrum" in plot_paths:
            img = self._embed_image(plot_paths["itpc_spectrum"], "DFT ITPC Frequency Spectrum")
            sections.append(
                "<h3>Cortical Tracking Frequency Spectrum (DFT)</h3>"
                f"<div class='plot-card'>{img}"
                "<figcaption>Channel-averaged ITPC across 0.5&ndash;4 Hz. "
                "Dashed lines mark word (3.125 Hz), phrase (1.56 Hz), sentence (0.78 Hz). "
                "Sharp peaks at target frequencies confirm stimulus-locked entrainment."
                "</figcaption></div>"
            )

        topo_html = ""
        for freq, label in _TARGET_FREQS:
            key = f"topomap_{label.lower()}"
            if key in plot_paths:
                img = self._embed_image(plot_paths[key], f"{label} Topomap")
                topo_html += (
                    f"<div class='plot-card'>{img}"
                    f"<figcaption>ITPC Topomap @ {freq} Hz ({label} rate). "
                    f"Left temporal/frontal hotspots (T7, F7) indicate expected language lateralization."
                    f"</figcaption></div>"
                )
        if topo_html:
            sections.append(f"<h3>ITPC Topographic Maps (DFT)</h3><div class='plot-grid'>{topo_html}</div>")

        if "morlet_tfr" in plot_paths:
            img = self._embed_image(plot_paths["morlet_tfr"], "Morlet TFR")
            sections.append(
                "<h3>Time-Frequency Representation (Morlet)</h3>"
                f"<div class='plot-card'>{img}"
                "<figcaption>ITPC time-frequency map averaged across channels. "
                "Shows temporal stability of phase-locking to the speech stimulus."
                "</figcaption></div>"
            )

        if "morlet_spectrum" in plot_paths:
            img = self._embed_image(plot_paths["morlet_spectrum"], "Morlet ITPC Frequency Spectrum")
            sections.append(
                "<h3>Cortical Tracking Frequency Spectrum (Morlet)</h3>"
                f"<div class='plot-card'>{img}"
                "<figcaption>Time-averaged Morlet ITPC across 0.5&ndash;4 Hz. "
                "Dashed lines mark word (3.125 Hz), phrase (1.56 Hz), sentence (0.78 Hz). "
                "Broader peaks than DFT reflect the Morlet wavelet's time-frequency trade-off."
                "</figcaption></div>"
            )

        morlet_topo_html = ""
        for freq, label in _TARGET_FREQS:
            key = f"morlet_topomap_{label.lower()}"
            if key in plot_paths:
                img = self._embed_image(plot_paths[key], f"Morlet {label} Topomap")
                morlet_topo_html += (
                    f"<div class='plot-card'>{img}"
                    f"<figcaption>Morlet ITPC Topomap @ {freq} Hz ({label} rate). "
                    "Spatial distribution is comparable to DFT; left temporal/frontal hotspots "
                    "(T7, F7) indicate expected language lateralization."
                    "</figcaption></div>"
                )
        if morlet_topo_html:
            sections.append(f"<h3>ITPC Topographic Maps (Morlet)</h3><div class='plot-grid'>{morlet_topo_html}</div>")

        return "\n".join(sections)

    def _build_legend_box(self) -> str:
        items = [
            {
                "term": "ITPC (Inter-Trial Phase Coherence)",
                "desc": (
                    "Ranges 0&ndash;1. Measures consistency of EEG phase across trials at a given frequency. "
                    "Values near 0 = random phase. Values near 1 = perfect phase-locking."
                ),
                "ranges": [
                    ("legend-excellent", "&gt; 0.15 &mdash; Strong entrainment"),
                    ("legend-good", "0.10&ndash;0.15 &mdash; Moderate"),
                    ("legend-ok", "0.06&ndash;0.10 &mdash; Weak"),
                    ("legend-bad", "&lt; 0.06 &mdash; Near chance"),
                ],
            },
            {
                "term": "itpc_comprehension_combined",
                "desc": (
                    "Average of phrase (1.56 Hz) and sentence (0.78 Hz) ITPC. "
                    "These rates have no acoustic correlate in the stimulus envelope &mdash; "
                    "entrainment here reflects top-down cognitive speech comprehension (Sokoliuk 2021)."
                ),
            },
            {
                "term": "ratio_cognitive_acoustic",
                "desc": (
                    "itpc_comprehension_combined / itpc_word. Separates bottom-up acoustic processing "
                    "(word rate, always present physically) from top-down linguistic comprehension."
                ),
                "ranges": [
                    ("legend-excellent", "&gt; 0.7 &mdash; Strong cognitive"),
                    ("legend-good", "0.5&ndash;0.7 &mdash; Moderate"),
                    ("legend-ok", "0.3&ndash;0.5 &mdash; Weak"),
                    ("legend-bad", "&lt; 0.3 &mdash; Predominantly acoustic"),
                ],
            },
            {
                "term": "Lateralization Index (LI)",
                "desc": (
                    "LI = (LH &minus; RH) / (LH + RH). Range: &minus;1 to +1. "
                    "Positive = left-dominant (expected for language in right-handed individuals). "
                    "LH: Fp1, F7, T7, F3, C3, P3. RH: Fp2, F8, T8, F4, C4, P4."
                ),
                "ranges": [
                    ("legend-excellent", "&gt; 0.1 &mdash; Left-lateralized"),
                    ("legend-ok", "&minus;0.1 to 0.1 &mdash; Bilateral"),
                    ("legend-bad", "&lt; &minus;0.1 &mdash; Right-lateralized (atypical)"),
                ],
            },
            {
                "term": "dft_p_comprehension",
                "desc": (
                    "One-sided permutation p-value (1000 surrogates, trial-level phase-scrambling). "
                    "Proportion of surrogates &ge; observed ITPC."
                ),
                "ranges": [
                    ("legend-excellent", "&lt; 0.01 &mdash; Highly significant"),
                    ("legend-good", "0.01&ndash;0.05 &mdash; Significant"),
                    ("legend-ok", "0.05&ndash;0.10 &mdash; Marginal"),
                    ("legend-bad", "&gt; 0.10 &mdash; Not significant"),
                ],
            },
        ]
        return style_utils.build_legend_box("Metrics Lexicon &amp; Interpretation", items)

    def build_session_html(self) -> str:
        """Return a collapsible <details> HTML fragment for combined multi-session reports."""
        plot_paths = self._save_plots()
        content = (
            self._build_overview_cards()
            + self._build_entrainment_table()
            + self._build_lateralization_section()
            + self._build_plots_section(plot_paths)
            + self._build_legend_box()
        )
        return (
            "<details class='session-wrapper' open>\n"
            f"{style_utils.build_session_panel(self.session_id, collapsible=True)}\n"
            f"<div class='session-content'>{content}</div>\n"
            "</details>\n"
        )

    def generate(self) -> Path:
        """Generate a standalone single-session HTML report and write to disk."""
        plot_paths = self._save_plots()
        pid = self.lt_obj.patient_id

        html = style_utils.build_html_header(
            title=f"Language Tracking Report \u2014 {pid} / {self.session_id}",
            patient_id=pid,
            session_id=self.session_id,
            extra_css=self._build_css_extensions(),
        )
        html += self._build_overview_cards()
        html += self._build_entrainment_table()
        html += self._build_lateralization_section()
        html += self._build_plots_section(plot_paths)
        html += self._build_legend_box()
        html += style_utils.build_html_footer("Language Tracking Pipeline")

        self.report_file.write_text(html, encoding="utf-8")
        logger.info(f"Report written to {self.report_file}")
        return self.report_file
