# Report Utilities — `src/reports/style_utils.py`

Shared HTML styling and component library used by all AwakenAI pipeline reports.
Nothing in this module is pipeline-specific — any new pipeline can import it
and compose a full HTML report without writing CSS or boilerplate.

---

## Public API

### Colors & constants

| Symbol | Value | Purpose |
|--------|-------|---------|
| `UW_PURPLE` | `#4b2e83` | Primary brand color |
| `WHITE` | `#ffffff` | Card / table backgrounds |
| `BG_MAIN` | `#fafafa` | Page background |
| `TEXT_MAIN` | `#1a1a1a` | Body text |
| `TEXT_MUTED` | `#333333` | Secondary text |
| `BORDER_LIGHT` | `#e0e0e0` | Table / card borders |
| `ROW_EVEN` | `#f5f0fb` | Alternating table rows |
| `BG_SUCCESS / TEXT_SUCCESS` | green pair | Positive states |
| `BG_DANGER / TEXT_DANGER` | red pair | Negative / alert states |
| `BG_WARNING / TEXT_WARNING` | yellow pair | Marginal states |
| `BG_INFO / TEXT_INFO` | blue pair | Informational states |
| `ICON_TRUE` | `✓` span | Green check for boolean tables |
| `ICON_FALSE` | `✗` span | Red cross for boolean tables |

---

### Functions

#### Document helpers

```python
build_html_header(title, patient_id=None, session_id=None, extra_css="") -> str
```
Returns the opening `<!DOCTYPE html> … <body>` block with all base CSS.
Optionally renders a dark-purple patient panel and/or a light-purple session
panel below the `<h1>` heading.

> **Multi-patient / multi-session reports:** pass neither `patient_id` nor
> `session_id` here. Instead, call `build_patient_panel(pid)` once per patient
> and `build_session_panel(sess)` (via `build_session_html()`) once per session
> inside your fragment loop, then stitch with `stitch_and_save()`.

```python
build_html_footer(generator_name) -> str
```
Closes `</body></html>` with a credit footer line.

```python
stitch_and_save(fragments, output_path, title="AwakenAI Report",
                generator_name="AwakenAI", extra_css="", pdf_path=None) -> Path
```
Wraps an ordered list of HTML fragment strings in a full document and writes
it to disk. Parent directories are created automatically.

If `pdf_path` is passed, the document embeds a "Download PDF" button at the top right with Javascript logic that `fetch()`es the expected PDF filename. If the PDF does not exist alongside the HTML, the button gracefully falls back to `window.print()`. Returns the resolved `Path` of the written HTML file.

---

#### Section panels

```python
build_patient_panel(patient_id) -> str
```
Dark purple full-width bar — use once per patient in a combined report.

```python
build_session_panel(session_id, collapsible=False) -> str
```
Light purple left-border bar — use once per session.  
When `collapsible=True`, renders as a `<summary>` element (for use inside a
`<details class="session-wrapper">` wrapper) with a rotating `⌄/⌃` arrow.

---

#### UI components

```python
build_status_badge(text, bg_color="#16a34a", text_color="#fff") -> str
```
Pill-shaped badge — e.g. `CMD+` in green, `CMD-` in red.

```python
build_metric_cards(cards: List[{"title", "value", "desc"}]) -> str
```
Responsive flex row of stat cards, each with a purple top border.

```python
build_metric_table(headers, rows, title="") -> str
```
Scrollable table wrapper with an optional `<h3>` heading above it.

```python
build_legend_box(title, items: List[{"term", "desc", "ranges"}]) -> str
```
Left-bordered definition list with optional color-coded range badges
(`legend-excellent`, `legend-good`, `legend-ok`, `legend-bad`).

```python
build_base_html_table(headers, rows, class_name="") -> str
```
Bare `<table>` element — use when you need full control over the wrapper.

---

#### PDF Export & CLI Utilities (`src/cli/cli_utils.py`)

HTML-to-PDF export is managed outside of `style_utils` to keep the UI layer clean and avoid circular dependencies:

```python
generate_pdf_from_html(html_path: Path, pdf_path: Path) -> bool
```

Uses headless Chromium via Playwright to render the HTML document and dump it as an A4 PDF (`print_background=True`). It ignores problematic OS-level C-dependencies by automatically downloading its own browser binary on first run. Returns `True` if generation succeeded.

```python
print_report_paths(pid: str, sess: str, html_path: Path, pdf_path: Path | None) -> None
```

Formats output paths using a terminal tree visual:

```
  CON001 / sess_01:
    └─ HTML: reports/command-following/CON001...html
    └─ PDF:  reports/command-following/CON001...pdf
```

---

## Usage patterns

### Standalone single-session report

```python
from src.reports import style_utils

html  = style_utils.build_html_header(
    "My Pipeline Report",
    patient_id="P001",
    session_id="sess_01",
    extra_css=my_extra_css,
)
html += my_body_html
html += style_utils.build_html_footer("My Pipeline")
Path("report.html").write_text(html, encoding="utf-8")
```

### Combined multi-patient / multi-session report

Build HTML fragments in a loop, then stitch once at the end.

```python
from src.reports import style_utils

fragments: list[str] = []
extra_css = ""

for pid in patient_ids:
    fragments.append(style_utils.build_patient_panel(pid))
    for sess in sessions:
        cf_report = MyPipelineReport(pipeline, session_id=sess)
        extra_css  = cf_report._build_css_extensions()     # capture once
        fragments.append(cf_report.build_session_html())    # collapsible panel + content

style_utils.stitch_and_save(
    fragments,
    output_path=output_dir / "combined_report.html",
    title="My Pipeline — Combined Report",
    generator_name="My Pipeline",
    extra_css=extra_css,
    pdf_path=output_dir / "combined_report.pdf",
)

# 2. Separately render the generated HTML to PDF
status = cli_utils.generate_pdf_from_html(
    output_dir / "combined_report.html",
    output_dir / "combined_report.pdf"
)

# 3. Print tree-formatted output
cli_utils.print_report_paths(
    "P001", "Combined",
    html_path=output_dir / "combined_report.html",
    pdf_path=output_dir / "combined_report.pdf" if status else None
)
```

`build_session_html()` is a convention every pipeline report class should
implement. It returns one self-contained fragment:

```html
<details class="session-wrapper" open>
  <summary class="session-header">Session: sess_01 ⌄</summary>
  <div class="session-content">… tables, plots, legend …</div>
</details>
```

Clicking the header bar collapses/expands the session content — no JavaScript,
pure CSS `<details>` behaviour.

---

## CSS classes provided by `render_uw_css()`

| Class                           | Element                | Purpose                                                    |
| ------------------------------- | ---------------------- | ---------------------------------------------------------- |
| `.table-wrapper`                | `<div>`                | Horizontal scroll container for wide tables                |
| `.metric-cards`                 | `<div>`                | Flex row of stat cards                                     |
| `.metric-card`                  | `<div>`                | Individual stat card                                       |
| `.metric-card-title/value/desc` | `<div>`                | Card sub-elements                                          |
| `.legend-box`                   | `<div>`                | Bordered definition list                                   |
| `.legend-range`                 | `<span>`               | Color-coded badge inside legend                            |
| `.legend-excellent/good/ok/bad` | modifier               | Semantic color variants                                    |
| `.session-card`                 | `<div>`                | Standalone session card (single-report use)                |
| `.session-wrapper`              | `<div>` or `<details>` | Spacing wrapper for combined report                        |
| `.session-content`              | `<div>`                | Card body inside a session wrapper                         |
| `.session-toggle-arrow`         | `<span>`               | Rotating chevron inside collapsible panel                  |
| `.html-btn`                     | `<button>`             | Base button styling for the report UI                      |
| `.html-btn-primary`             | modifier               | Purple-stroked primary button styling                      |
| `.html-btn-download`            | modifier               | Absolute positioning for the top-right PDF download button |
