"""Utility functions for HTML report styling and generation across AwakenAI pipelines."""

from pathlib import Path
from typing import Any, Dict, List

UW_PURPLE = "#4b2e83"
TEXT_MAIN = "#1a1a1a"  # Dark Gray for main text
BG_MAIN = "#fafafa"  # Off-White background
WHITE = "#ffffff"
SHADOW_LIGHT = "rgba(0,0,0,0.1)"  # Light shadow
BORDER_LIGHT = "#e0e0e0"  # Light gray border
ROW_EVEN = "#f5f0fb"  # Very light purple for even table rows
ROW_HOVER = "#e8dff5"  # Light purple for table row hover
LEGEND_BG = "#f9f7fd"  # Very light purple for legend background
TEXT_MUTED = "#333333"  # Dark gray for muted text
FOOTER_TEXT = "#888888"  # Gray for footer text
FOOTER_BORDER = "#dddddd"  # Light gray for footer border

# Status colors
BG_SUCCESS = "#d4edda"  # Light Green
TEXT_SUCCESS = "#155724"  # Dark Green
BG_INFO = "#cce5ff"  # Light Blue
TEXT_INFO = "#004085"  # Dark Blue
BG_WARNING = "#fff3cd"  # Light Yellow
TEXT_WARNING = "#856404"  # Dark Yellow
BG_DANGER = "#f8d7da"  # Light Red
TEXT_DANGER = "#721c24"  # Dark Red

# Card colors
CARD_BORDER = "#e2e8f0"  # Light gray/blue border
CARD_TEXT_MUTED = "#64748b"  # Slate gray text
CARD_TEXT_DARK = "#1e293b"  # Dark slate text

# Semantic boolean icons — use instead of plain True/False in tables
ICON_TRUE = "<span style='color:#16a34a;font-size:1rem;' title='True'>&#10003;</span>"
ICON_FALSE = "<span style='color:#dc2626;font-size:1rem;' title='False'>&#10007;</span>"


def render_uw_css() -> str:
    """Inline CSS for a clean, readable report matching the UW visual identity."""
    return f"""
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 2rem auto;
        max-width: 1400px;
        padding: 0 1.5rem;
        color: {TEXT_MAIN};
        background: {BG_MAIN};
        line-height: 1.5;
    }}
    h1 {{ color: {UW_PURPLE}; border-bottom: 2px solid {UW_PURPLE}; padding-bottom: 0.3rem; }}
    h2, h3, h4 {{ color: {UW_PURPLE}; margin-top: 2rem; }}
    
    .table-wrapper {{
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        margin: 1rem 0;
    }}
    table {{
        border-collapse: collapse;
        min-width: 600px;
        width: 100%;
        background: {WHITE};
        box-shadow: 0 1px 3px {SHADOW_LIGHT};
        font-size: 0.875rem;
    }}
    th, td {{
        padding: 0.45rem 0.75rem;
        text-align: left;
        border-bottom: 1px solid {BORDER_LIGHT};
        white-space: nowrap;
    }}
    th {{ 
        background: {UW_PURPLE}; color: {WHITE}; font-weight: 600; 
        text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.05em; 
    }}
    tbody tr:nth-child(even) {{ background: {ROW_EVEN}; }}
    tbody tr:hover {{ background: {ROW_HOVER}; }}
    
    .legend-box {{ font-size: 0.8rem; color: {TEXT_MUTED}; background: {LEGEND_BG}; border-left: 3px solid {UW_PURPLE};
                  padding: 0.6rem 1rem; margin: 0.5rem 0 1.2rem; line-height: 1.7; }}
    .legend-box dt {{ font-weight: 700; color: {UW_PURPLE}; margin-top: 0.35rem; }}
    .legend-box dd {{ margin: 0 0 0 1rem; }}
    .legend-range {{ display: inline-block; padding: 0.1rem 0.45rem; border-radius: 3px;
                    font-size: 0.75rem; font-weight: 600; margin-top: 0.25rem; }}
                    
    .legend-excellent {{ background: {BG_SUCCESS}; color: {TEXT_SUCCESS}; }}
    .legend-good      {{ background: {BG_INFO}; color: {TEXT_INFO}; }}
    .legend-ok        {{ background: {BG_WARNING}; color: {TEXT_WARNING}; }}
    .legend-bad       {{ background: {BG_DANGER}; color: {TEXT_DANGER}; }}
    
    .metric-cards {{
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }}
    .metric-card {{
        flex: 1;
        min-width: 250px;
        background: {WHITE};
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px {SHADOW_LIGHT};
        border-top: 4px solid {UW_PURPLE};
    }}
    .metric-card-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {CARD_TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}
    .metric-card-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {CARD_TEXT_DARK};
        margin-bottom: 0.25rem;
    }}
    .metric-card-desc {{
        font-size: 0.85rem;
        color: {CARD_TEXT_MUTED};
    }}
    
    .session-card {{
        background: {WHITE};
        border-radius: 8px;
        box-shadow: 0 4px 6px {SHADOW_LIGHT};
        border: 1px solid {BORDER_LIGHT};
        padding: 2.5rem;
        margin-top: 2rem;
    }}

    /* Multi-session combined report */
    .session-wrapper {{
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }}
    .session-content {{
        background: {WHITE};
        border: 1px solid {BORDER_LIGHT};
        border-top: none;
        border-radius: 8px;
        box-shadow: 0 2px 6px {SHADOW_LIGHT};
        padding: 1.5rem;
    }}

    /* Collapsible session panel */
    details.session-wrapper > summary {{
        list-style: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    details.session-wrapper > summary::-webkit-details-marker {{ display: none; }}
    details.session-wrapper > summary::marker {{ display: none; }}
    .session-toggle-arrow {{
        font-size: 1.0rem;
        color: {TEXT_MUTED};
        transition: transform 0.35s ease;
        user-select: none;
        padding: 0 0.25rem;
    }}
    details.session-wrapper[open] .session-toggle-arrow {{
        transform: rotate(180deg);
    }}
    """


def build_base_html_table(headers: List[str], rows: List[Any], class_name: str = "") -> str:
    """Build a simple HTML <table> from headers and rows."""
    cls_attr = f" class='{class_name}'" if class_name else ""
    parts = [f"<table{cls_attr}>\n<thead>\n<tr>"]
    for h in headers:
        parts.append(f"<th>{h}</th>")
    parts.append("</tr>\n</thead>\n<tbody>\n")
    for row in rows:
        parts.append("<tr>")
        cells = row if isinstance(row, (list, tuple)) else list(row)
        for cell in cells:
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>\n")
    parts.append("</tbody>\n</table>")
    return "".join(parts)


def build_patient_panel(patient_id: str) -> str:
    """Dark purple full-width bar used as a patient section header."""
    return (
        f"<div class='patient-header' style='background-color:{UW_PURPLE};color:{WHITE};"
        f"padding:0.75rem 1.5rem;border-radius:8px;margin-top:1.5rem;'>"
        f"<h2 style='margin:0;font-size:1.5rem;color:{WHITE};'>Patient: {patient_id}</h2>"
        f"</div>"
    )


def build_session_panel(session_id: str, collapsible: bool = False) -> str:
    """Light purple left-border bar used as a session section header.

    Args:
        session_id: Session identifier to display.
        collapsible: When True, renders as a ``<summary>`` element for use
            inside a ``<details class='session-wrapper'>`` wrapper. Adds a
            rotating down/up arrow on the right to signal toggle state.
            When False (default), renders as a plain ``<div>``.
    """
    shared_style = (
        f"background-color:{ROW_EVEN};padding:0.75rem 1.5rem;"
        f"border-left:4px solid {UW_PURPLE};border-top:1px solid {BORDER_LIGHT};"
        f"border-right:1px solid {BORDER_LIGHT};margin-top:0;"
        f"border-radius:0;"
    )
    label = f"<h3 style='margin:0;font-size:1.1rem;color:{UW_PURPLE};'>Session: {session_id}</h3>"

    if collapsible:
        arrow = "<span class='session-toggle-arrow'>&#8964;</span>"
        return f"<summary class='session-header' style='{shared_style}'>{label}{arrow}</summary>"
    return f"<div class='session-header' style='{shared_style}'>{label}</div>"


def stitch_and_save(
    fragments: List[str],
    output_path: Path,
    title: str = "AwakenAI Report",
    generator_name: str = "AwakenAI",
    extra_css: str = "",
) -> Path:
    """Wrap a list of HTML fragment strings in a full document and write to disk.

    Intended for the multi-patient / multi-session combined report flow:
    the CLI runner accumulates patient panels and session fragments into a list,
    then calls this once to produce a single HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html_header(title, extra_css=extra_css)
    html += "\n".join(fragments)
    html += build_html_footer(generator_name)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_html_header(title: str, patient_id: str = None, session_id: str = None, extra_css: str = "") -> str:
    """Builds the common HTML head and page header with consistent styling."""

    patient_session_html = ""
    if patient_id or session_id:
        patient_html = build_patient_panel(patient_id) if patient_id else ""
        session_html = build_session_panel(session_id) if session_id else ""
        patient_session_html = patient_html + session_html

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        {render_uw_css()}
        {extra_css}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>{patient_session_html}
    </div>
"""


def build_html_footer(generator_name: str) -> str:
    """Closes the HTML document with a common footer."""
    return f"""
    <footer style="margin-top: 3rem; color: {FOOTER_TEXT}; font-size: 0.85rem; 
                   border-top: 1px solid {FOOTER_BORDER}; padding-top: 1rem; margin-bottom: 2rem;">
        <p>Report generated by {generator_name} &mdash; AwakenAI Capstone</p>
    </footer>
</body>
</html>
"""


def build_status_badge(text: str, bg_color: str = "#16a34a", text_color: str = "#fff") -> str:
    """Pill-shaped status badge (e.g. CMD+, CMD-)."""
    return (
        f"<span style='background:{bg_color};color:{text_color};padding:0.3rem 0.9rem;"
        f"border-radius:999px;font-size:1.1rem;font-weight:bold;'>{text}</span>"
    )


def build_metric_cards(cards: List[Dict[str, str]]) -> str:
    """Builds a row of metric cards. Each card is a dict with 'title', 'value', 'desc'."""
    html = "<div class='metric-cards'>\n"
    for c in cards:
        html += f"""
        <div class="metric-card">
            <div class="metric-card-title">{c.get("title", "")}</div>
            <div class="metric-card-value">{c.get("value", "")}</div>
            <div class="metric-card-desc">{c.get("desc", "")}</div>
        </div>
        """
    html += "</div>\n"
    return html


def build_metric_table(headers: List[str], rows: List[Any], title: str = "") -> str:
    """Builds a table for metrics."""
    title_html = f"<h3>{title}</h3>\n" if title else ""
    return f"{title_html}<div class='table-wrapper'>\n{build_base_html_table(headers, rows)}\n</div>"


def build_legend_box(title: str, items: List[Dict[str, Any]]) -> str:
    """
    Builds a legend box with terms, descriptions, and optional color-coded range badges.
    Each item in `items` should be a dict:
      {
          "term": "erd_dB",
          "desc": "Mean Event...",
          "ranges": [
              ("legend-excellent", "< -2 dB — Strong"),
              ("legend-good", "-1 to -2 dB — Good")
          ]
      }
    """
    html = f"""
    <div class="legend-box">
        <p style="margin-top: 0; font-weight: bold; color: {UW_PURPLE};">{title}</p>
        <dl>
"""
    for item in items:
        html += f"            <dt>{item['term']}</dt>\n"
        html += f"            <dd>{item['desc']}\n"
        if item.get("ranges"):
            html += "            <br/>\n"
            for r_class, r_text in item["ranges"]:
                html += f'            <span class="legend-range {r_class}">{r_text}</span>\n'
        html += "            </dd>\n"
    html += """
        </dl>
    </div>
"""
    return html
