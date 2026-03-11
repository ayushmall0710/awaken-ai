"""HTML report for Language Tracking Analysis."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import src.reports.style_utils as style_utils
from src.data_loading import config
from src.pipelines.language_tracking import LanguageConfig
from src.viz.language_plots import (
    plot_focus_comparison_bar,
    plot_itpc_channel_bar,
    plot_itpc_channels_horizontal,
    plot_itpc_results,
    plot_itpc_spectrum,
    plot_itpc_topomap,
)

logger = logging.getLogger(__name__)

# Use default config for static labels/metrics
_DEFAULT_CFG = LanguageConfig()

_ENTRAINMENT_COLS = [
    ("Focus", "focus"),
    (f"Word ({_DEFAULT_CFG.target_word_freq} Hz)", "itpc_word"),
    (f"Phrase ({_DEFAULT_CFG.target_phrase_freq} Hz)", "itpc_phrase"),
    (f"Sentence ({_DEFAULT_CFG.target_sentence_freq} Hz)", "itpc_sentence"),
    ("Comprehension", "itpc_comprehension"),
    ("Ratio S/W", "ratio_sent_word"),
    ("Ratio S/P", "ratio_sent_phrase"),
    ("Ratio BW", "ratio_bw_normalized"),
]


_TARGET_FREQS = [
    (_DEFAULT_CFG.target_word_freq, "Word"),
    (_DEFAULT_CFG.target_phrase_freq, "Phrase"),
    (_DEFAULT_CFG.target_sentence_freq, "Sentence"),
]

_MORLET_METRICS = {
    "freq_sentence_hz": _DEFAULT_CFG.target_sentence_freq,
    "freq_phrase_hz": _DEFAULT_CFG.target_phrase_freq,
    "freq_word_hz": _DEFAULT_CFG.target_word_freq,
}

_MORLET_COLS = [
    ("Focus", "focus"),
    ("Word (Morlet)", "morlet_itpc_word"),
    ("Phrase (Morlet)", "morlet_itpc_phrase"),
    ("Sentence (Morlet)", "morlet_itpc_sentence"),
    ("Comprehension (Morlet)", "morlet_itpc_comprehension"),
]


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
        self._summary: Optional[dict] = None

    def _get_summary(self) -> dict:
        if self._summary is None:
            self._summary = self.lt_obj.generate_summary()
        return self._summary

    def _save_plots(self) -> dict:
        """Generate all plots to output_dir. Returns {key: Path}. Result is cached."""
        if self._plot_paths is not None:
            return self._plot_paths

        res = self.lt_obj.results
        pid = self.lt_obj.patient_id
        spectrum_full = self.lt_obj._dft_spectrum_full
        freqs_full = self.lt_obj._dft_freqs
        info_full = self.lt_obj._dft_info
        paths = {}

        # 1. Focus Comparison Bar Plot
        paths["focus_comparison"] = plot_focus_comparison_bar(res, pid, self.output_dir)

        # 2. Per-Focus Plots (Clinical, LH, RH, and Optimal)
        cfg = getattr(self.lt_obj, "cfg", _DEFAULT_CFG)
        word_idx = int(np.argmin(np.abs(freqs_full - cfg.target_word_freq)))
        vmax = max(float(np.percentile(spectrum_full[:, word_idx], 95)) * 1.2, 0.1)
        vlim = (0.0, vmax)

        for focus in ["clinical", "lh", "rh", "optimal"]:
            rows = res[res["focus"] == focus]
            if rows.empty:
                continue
            row = rows.iloc[0]
            channels = row.get("channels")
            if not isinstance(channels, (list, tuple, np.ndarray)) or len(channels) == 0:
                continue

            metrics = row.to_dict()
            focus_label = focus.capitalize()

            # Map channel names to indices
            ch_to_idx = {ch: i for i, ch in enumerate(self.lt_obj._dft_ch_names)}
            ch_indices = [ch_to_idx[ch] for ch in channels if ch in ch_to_idx]
            if not ch_indices:
                continue

            # Subset spectrum for this focus
            focus_spectrum = spectrum_full[ch_indices, :]

            # Spectrum Plot
            paths[f"itpc_spectrum_{focus}"] = plot_itpc_spectrum(
                focus_spectrum, freqs_full, pid, self.output_dir, metrics, focus_label=focus_label
            )

            # Topomaps (at target frequencies)
            # Use full spectrum but highlight focus channels (except for Word rate in Optimal focus)
            for freq, label in _TARGET_FREQS:
                highlight = None
                if focus == "optimal" and label != "Word":
                    highlight = channels

                paths[f"topomap_{label.lower()}_{focus}"] = plot_itpc_topomap(
                    spectrum_full,
                    freqs_full,
                    info_full,
                    freq,
                    label,
                    pid,
                    self.output_dir,
                    vlim=vlim,
                    highlight_channels=highlight,
                )

        # 3. Morlet Plots (Secondary validation, usually Clinical focus only)
        if self.lt_obj._morlet_itc is not None:
            try:
                clinical_row = res[res["focus"] == "clinical"].iloc[0]
                paths["morlet_tfr"] = plot_itpc_results(
                    self.lt_obj._morlet_itc, pid, str(self.output_dir), _MORLET_METRICS
                )

                morlet_data = self.lt_obj._morlet_itc.data  # (n_channels, n_freqs, n_times)
                paths["morlet_channel_bar"] = plot_itpc_channel_bar(
                    morlet_data,
                    list(self.lt_obj._dft_ch_names),
                    pid,
                    str(self.output_dir),
                    int(clinical_row["n_trials"]),
                    self.lt_obj._morlet_itc.freqs,
                    cfg.sentence_band,
                    cfg.phrase_band,
                    cfg.word_band,
                )
                morlet_spectrum = np.mean(morlet_data, axis=-1)  # (n_channels, n_freqs)
                morlet_freqs = self.lt_obj._morlet_itc.freqs

                morlet_word_idx = int(np.argmin(np.abs(morlet_freqs - cfg.target_word_freq)))
                morlet_vmax = max(float(np.percentile(morlet_spectrum[:, morlet_word_idx], 95)) * 1.2, 0.1)
                morlet_vlim = (0.0, morlet_vmax)

                for freq, label in _TARGET_FREQS:
                    paths[f"morlet_topomap_{label.lower()}"] = plot_itpc_topomap(
                        morlet_spectrum,
                        morlet_freqs,
                        info_full,
                        freq,
                        label,
                        pid,
                        self.output_dir,
                        vlim=morlet_vlim,
                        method_label="Morlet",
                    )
            except Exception as e:
                logger.warning(f"Morlet plots failed: {e}")

        # DFT channel bar (Horizontal) - Always generate if results are available
        try:
            clinical_rows = res[res["focus"] == "clinical"]
            if not clinical_rows.empty:
                clinical_row = clinical_rows.iloc[0]
                paths["dft_channel_bar"] = plot_itpc_channels_horizontal(
                    spectrum_full,
                    list(self.lt_obj._dft_ch_names),
                    pid,
                    str(self.output_dir),
                    int(clinical_row["n_trials"]),
                    freqs_full,
                    cfg.sentence_band,
                    cfg.phrase_band,
                    cfg.word_band,
                    method_label="DFT",
                )
        except Exception as e:
            logger.warning(f"DFT channel bar plot failed: {e}")

        self._plot_paths = paths
        return paths

    def _format_cell(self, key: str, row: pd.Series) -> str:
        val = row.get(key)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "N/A"

        # Dictionary of ITPC keys to their corresponding p-value keys
        itpc_p_map = {
            "itpc_word": "dft_p_word",
            "itpc_phrase": "dft_p_phrase",
            "itpc_sentence": "dft_p_sentence",
            "itpc_comprehension": "dft_p_comprehension",
            "morlet_itpc_word": "morlet_p_word",
            "morlet_itpc_phrase": "morlet_p_phrase",
            "morlet_itpc_sentence": "morlet_p_sentence",
            "morlet_itpc_comprehension": "morlet_p_comprehension",
            "lh_itpc_word": "lh_p_word",
            "lh_itpc_phrase": "lh_p_phrase",
            "lh_itpc_sentence": "lh_p_sentence",
            "lh_itpc_comprehension": "lh_p_comprehension",
            "rh_itpc_word": "rh_p_word",
            "rh_itpc_phrase": "rh_p_phrase",
            "rh_itpc_sentence": "rh_p_sentence",
            "rh_itpc_comprehension": "rh_p_comprehension",
        }

        if key in itpc_p_map:
            p_key = itpc_p_map[key]
            p_val = row.get(p_key)
            if p_val is not None and np.isfinite(p_val):
                return style_utils.format_with_significance(float(val), float(p_val))
            return f"{float(val):.4f}"

        if key in (
            "ratio_sent_word",
            "ratio_sent_phrase",
            "ratio_bw_normalized",
        ):
            return f"{float(val):.4f}"

        if key.startswith("lateralization_index_"):
            v = float(val)
            color = "#16a34a" if v > 0.1 else ("#dc2626" if v < -0.1 else "#856404")
            return f"<span style='color:{color};font-weight:bold;'>{v:+.3f}</span>"
        return str(val)

    def _build_overview_cards(self) -> str:
        res = self.lt_obj.results
        clinical_row = res[res["focus"] == "clinical"].iloc[0]
        summary = self._get_summary()

        n_trials = int(clinical_row.get("n_trials", 0))
        itpc_comp = float(clinical_row.get("itpc_comprehension", 0))
        p_comp = float(clinical_row.get("dft_p_comprehension", 1.0))
        li_comp = float(summary.get("lateralization_index_comprehension", 0.0))

        # Check for significance in Clinical, LH, RH, or Optimal focuses
        significant_focuses = []
        for focus in ["clinical", "lh", "rh", "optimal"]:
            rows = res[res["focus"] == focus]
            if not rows.empty:
                p_val = float(rows.iloc[0].get("dft_p_comprehension", 1.0))
                if p_val < 0.05:
                    significant_focuses.append(focus.upper())

        if significant_focuses:
            focus_str = ", ".join(significant_focuses)
            badge = style_utils.build_status_badge("Significant", bg_color="#16a34a")
            badge_desc = f"Permutation test at phrase+sentence rates<br><strong>Significant:</strong> {focus_str}"
        else:
            badge = style_utils.build_status_badge("Not Significant", bg_color="#dc2626")
            badge_desc = "Permutation test at phrase+sentence rates"

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
                    "desc": badge_desc,
                },
                {
                    "title": "Lateralization",
                    "value": li_badge,
                    "desc": "No handedness assumed — interpret with clinical context",
                },
            ]
        )

    def _build_entrainment_table(self) -> str:
        headers = [name for name, _ in _ENTRAINMENT_COLS]
        table_rows = []
        for _, row in self.lt_obj.results.iterrows():
            cells = [self._format_cell(key, row) for _, key in _ENTRAINMENT_COLS]
            table_rows.append(cells)
        table = style_utils.build_base_html_table(headers, table_rows)
        return f"<h3>Linguistic Entrainment Metrics (DFT)</h3><div class='table-wrapper'>{table}</div>"

    def _build_optimal_focus_section(self, plot_paths: dict) -> str:
        res = self.lt_obj.results
        opt_rows = res[res["focus"] == "optimal"]
        if opt_rows.empty:
            return ""

        row = opt_rows.iloc[0]
        channels = row.get("channels")
        p_comp = row.get("dft_p_comprehension")

        # Focus Comparison Plot
        comp_html = ""
        if "focus_comparison" in plot_paths:
            img = style_utils.embed_image(plot_paths["focus_comparison"], "Focus Comparison")
            comp_html = (
                "<div style='flex: 1.2; min-width: 350px; max-width: 500px;'>"
                f"<div class='plot-card' style='padding: 0.5rem;'>{img}"
                "<figcaption style='font-size: 0.7rem;'>Comprehension ITPC compared across "
                "Clinical, LH, RH, and Optimal focuses. "
                "Stars indicate significance levels.</figcaption></div>"
                "</div>"
            )

        if not isinstance(channels, (list, tuple, np.ndarray)) or len(channels) == 0:
            text_content = (
                "<div style='flex: 1.5; min-width: 300px;'>"
                "<h3>Optimal Focus Identification</h3>"
                "<p>No significant spatial cluster was identified during this analysis session.</p>"
                "</div>"
            )
        else:
            ch_list = ", ".join(channels)
            badge = (
                style_utils.build_status_badge("Significant", bg_color="#16a34a")
                if p_comp < 0.05
                else style_utils.build_status_badge("Not Significant", bg_color="#dc2626")
            )
            text_content = (
                "<div style='flex: 1.5; min-width: 300px;'>"
                "<h3>Optimal Focus Identification</h3>"
                "<p>A data-driven optimal focus was identified using <strong>Cluster-informed Peak Selection</strong>. "
                "This logic first identifies statistically significant neural regions (clusters) across the scalp "
                "and then selects the Top 3 electrodes with the strongest comprehension-rate tracking from clusters. "
                "This ensures the focus is both statistically sound and representative of the peak neural response.</p>"
                f"<p><strong>Status:</strong> {badge} (p={p_comp:.3f})</p>"
                f"<p><strong>Electrodes:</strong> {ch_list}</p>"
                "</div>"
            )

        content = (
            "<div style='display: flex; flex-wrap: wrap; gap: 2rem; align-items: start;'>"
            f"{text_content}"
            f"{comp_html}"
            "</div>"
        )

        return style_utils.build_collapsible_panel("Optimal Focus & Comparison", content, open_default=True)

    def _build_morlet_section(self) -> str:
        headers = [name for name, _ in _MORLET_COLS]
        table_rows = []
        for _, row in self.lt_obj.results.iterrows():
            cells = [self._format_cell(key, row) for _, key in _MORLET_COLS]
            table_rows.append(cells)
        table = style_utils.build_base_html_table(headers, table_rows)
        note = (
            "<p style='font-size:0.8rem;color:#64748b;margin-top:0.5rem;'>"
            "Morlet ITPC is computed from time-averaged complex wavelets at each target frequency "
            "(phase-coherence across trials). Values are not numerically comparable to DFT ITPC. "
            "Morlet p-values are a secondary validation; DFT remains the primary significance measure "
            "per Sokoliuk 2021."
            "</p>"
        )
        return f"<h3>Morlet Wavelet Validation (All Focuses)</h3><div class='table-wrapper'>{table}</div>{note}"

    def _build_lateralization_section(self) -> str:
        summary = self._get_summary()
        # Convert summary dict to Series so _format_cell can access multiple keys
        summary_ser = pd.Series(summary)

        rates = [
            (
                f"Word ({_DEFAULT_CFG.target_word_freq} Hz)",
                "lh_itpc_word",
                "rh_itpc_word",
                "lateralization_index_word",
            ),
            (
                f"Phrase ({_DEFAULT_CFG.target_phrase_freq} Hz)",
                "lh_itpc_phrase",
                "rh_itpc_phrase",
                "lateralization_index_phrase",
            ),
            (
                f"Sentence ({_DEFAULT_CFG.target_sentence_freq} Hz)",
                "lh_itpc_sentence",
                "rh_itpc_sentence",
                "lateralization_index_sentence",
            ),
            ("Comprehension", "lh_itpc_comprehension", "rh_itpc_comprehension", "lateralization_index_comprehension"),
        ]
        headers = [
            "Rate",
            "LH ITPC (Fp1/F7/T7/F3/C3/P3)",
            "RH ITPC (Fp2/F8/T8/F4/C4/P4)",
            "Lateralization Index",
        ]
        table_rows = []
        for label, lh_key, rh_key, li_key in rates:
            lh_val = self._format_cell(lh_key, summary_ser) if lh_key else "&mdash;"
            rh_val = self._format_cell(rh_key, summary_ser) if rh_key else "&mdash;"
            li_val = self._format_cell(li_key, summary_ser)
            table_rows.append([label, lh_val, rh_val, li_val])

        table = style_utils.build_base_html_table(headers, table_rows)
        note = (
            "<p style='font-size:0.8rem;color:#64748b;margin-top:0.5rem;'>"
            "<strong>Note:</strong> Values derived from Discrete Fourier Transform (DFT) analysis. "
            "LI = (LH &minus; RH) / (LH + RH). Positive = left-dominant. "
            "No handedness assumption is made &mdash; interpret alongside clinical context."
            "</p>"
        )
        return f"<h3>Hemisphere Lateralization Analysis</h3><div class='table-wrapper'>{table}</div>{note}"

    def _build_plots_section(self, plot_paths: dict) -> str:
        sections = []

        # 2. Per-Focus Details (Unified Spectrum + Topomaps)
        sections.append("<h2>Linguistic Tracking Visualizations</h2>")
        focus_labels = {
            "clinical": "Standard 10-20 Clinical Focus",
            "lh": "Left Hemisphere (LH) Focus",
            "rh": "Right Hemisphere (RH) Focus",
            "optimal": "Data-Driven Optimal Focus",
        }

        for focus in ["clinical", "lh", "rh", "optimal"]:
            spec_key = f"itpc_spectrum_{focus}"
            if spec_key not in plot_paths:
                continue

            # Build Spectrum Card (Full Width)
            spec_img = style_utils.embed_image(plot_paths[spec_key], f"{focus.capitalize()} Spectrum")
            spec_desc = (
                f"<strong>Frequency Spectrum ({focus.upper()} focus):</strong> "
                "Channel-averaged ITPC vs Frequency. Vertical lines mark target Word (3.125 Hz), "
                "Phrase (1.56 Hz), and Sentence (0.78 Hz) rates. Peaks confirm stimulus-locked entrainment."
            )
            spec_html = style_utils.build_plot_card(spec_img, spec_desc)

            # Build Topo Grid (3 columns horizontally)
            topo_cards = []
            targets = [
                (_DEFAULT_CFG.target_word_freq, "Word"),
                (_DEFAULT_CFG.target_phrase_freq, "Phrase"),
                (_DEFAULT_CFG.target_sentence_freq, "Sentence"),
            ]

            # Use same order as in _save_plots to match keys
            for i, (freq, label) in enumerate(targets):
                key = f"topomap_{label.lower()}_{focus}"
                if key in plot_paths:
                    img = style_utils.embed_image(plot_paths[key], f"{label} Topomap ({focus})")
                    desc = f"<strong>{label} Rate</strong> ({freq} Hz)"
                    topo_cards.append(style_utils.build_plot_card(img, desc))

            topo_html = f"<div class='topo-grid'>{''.join(topo_cards)}</div>"
            topo_caption = (
                "<p class='plot-caption-muted'>"
                "<strong>Topographic ITPC Maps:</strong> Spatial distribution of phase-locking at target frequencies. "
                "All maps in this row share a unified scale for direct comparison. "
                f"{'White markers highlight the data-driven optimal cluster.' if focus == 'optimal' else ''}"
                "</p>"
            )

            # Combine: Spectrum followed by Topo row
            grid_content = f"{spec_html}{topo_html}{topo_caption}"

            # Wrap in Collapsible Panel
            panel_title = focus_labels.get(focus, focus.capitalize())
            sections.append(
                style_utils.build_collapsible_panel(panel_title, grid_content, open_default=(focus == "clinical"))
            )

        # 3. Validation & Per-Channel Analysis
        validation_html = []
        if "morlet_tfr" in plot_paths:
            img = style_utils.embed_image(plot_paths["morlet_tfr"], "Morlet TFR")
            desc = (
                "ITPC time-frequency map averaged across channels (Clinical focus). "
                "Shows temporal stability of phase-locking to the speech stimulus."
            )
            validation_html.append(
                "<h3>Time-Frequency Representation (Morlet)</h3>" + style_utils.build_plot_card(img, desc)
            )

        # Side-by-side Channel Bar Charts
        bar_cards = []
        if "dft_channel_bar" in plot_paths:
            img = style_utils.embed_image(plot_paths["dft_channel_bar"], "DFT Per-Channel ITPC")
            desc = "DFT Per-Channel ITPC (Horizontal). Dashed line = chance level 1/&radic;N."
            bar_cards.append(style_utils.build_plot_card(img, desc))

        if "morlet_channel_bar" in plot_paths:
            img = style_utils.embed_image(plot_paths["morlet_channel_bar"], "Morlet Per-Channel ITPC")
            desc = "Morlet Band-Averaged Per-Channel ITPC. Dashed line = chance level 1/&radic;N."
            bar_cards.append(style_utils.build_plot_card(img, desc))

        if bar_cards:
            validation_html.append("<h3>Per-Channel Validation (DFT vs Morlet)</h3>")
            validation_html.append(f"<div class='plot-grid'>{''.join(bar_cards)}</div>")

        morlet_topo_html = ""
        for freq, label in _TARGET_FREQS:
            key = f"morlet_topomap_{label.lower()}"
            if key in plot_paths:
                img = style_utils.embed_image(plot_paths[key], f"Morlet {label} Topomap")
                desc = (
                    f"Morlet ITPC Topomap @ {freq} Hz ({label} rate). "
                    "Left temporal/frontal hotspots indicate expected language lateralization."
                )
                morlet_topo_html += style_utils.build_plot_card(img, desc)

        if morlet_topo_html:
            validation_html.append(
                f"<h3>ITPC Topographic Maps (Morlet)</h3><div class='plot-grid'>{morlet_topo_html}</div>"
            )

        if validation_html:
            sections.append("<h2>Validation & Per-Channel Analysis</h2>" + "\n".join(validation_html))

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
                "term": "itpc_comprehension",
                "desc": (
                    f"Average of phrase ({_DEFAULT_CFG.target_phrase_freq} Hz) and "
                    f"sentence ({_DEFAULT_CFG.target_sentence_freq} Hz) ITPC. "
                    "These rates have no acoustic correlate in the stimulus envelope &mdash; "
                    "entrainment here reflects top-down cognitive speech comprehension (Sokoliuk 2021)."
                ),
            },
            {
                "term": "ratio_sent_word",
                "desc": (
                    "itpc_sentence / itpc_word. Measures the strength of sentence-level tracking relative "
                    "to bottom-up word tracking. Higher values indicate stronger linguistic integration."
                ),
            },
            {
                "term": "ratio_bw_normalized",
                "desc": (
                    "Combined Phrase+Sentence ITPC divided by Word ITPC. Specifically isolates "
                    "top-down linguistic tracking from acoustic tracking."
                ),
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
                "term": "DFT vs. Morlet Analysis",
                "desc": (
                    "<strong>Discrete Fourier Transform (DFT):</strong> Primary significance measure (Sokoliuk 2021). "
                    "Precisely isolates target frequencies using long-window transforms. "
                    "<br><strong>Morlet Wavelets:</strong> Secondary validation. Provides time-frequency resolution "
                    "to confirm stability of phase-locking across the entire stimulus duration."
                ),
            },
            {
                "term": "Significance Indicators",
                "desc": (
                    "ITPC values are annotated with stars to indicate statistical significance "
                    "from permutation testing (1000 surrogates)."
                ),
                "ranges": [
                    ("legend-excellent", "*** &mdash; p < 0.001"),
                    ("legend-good", "** &mdash; p < 0.01"),
                    ("legend-ok", "* &mdash; p < 0.05"),
                    ("legend-bad", "(None) &mdash; Not significant (p >= 0.05)"),
                ],
            },
            {
                "term": "dft_p_comprehension",
                "desc": (
                    "One-sided permutation p-value (1000 surrogates, trial-level phase-scrambling). "
                    "Proportion of surrogates &ge; observed ITPC."
                ),
            },
        ]
        return style_utils.build_legend_box("Metrics Lexicon &amp; Interpretation", items)

    def build_session_html(self) -> str:
        """Return a collapsible <details> HTML fragment for combined multi-session reports."""
        plot_paths = self._save_plots()
        content = (
            self._build_overview_cards()
            + self._build_entrainment_table()
            + self._build_optimal_focus_section(plot_paths)
            + self._build_morlet_section()
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
        )
        html += self._build_overview_cards()
        html += self._build_entrainment_table()
        html += self._build_optimal_focus_section(plot_paths)
        html += self._build_morlet_section()
        html += self._build_lateralization_section()
        html += self._build_plots_section(plot_paths)
        html += self._build_legend_box()
        html += style_utils.build_html_footer("Language Tracking Pipeline")

        self.report_file.write_text(html, encoding="utf-8")
        logger.info(f"Report written to {self.report_file}")
        return self.report_file
