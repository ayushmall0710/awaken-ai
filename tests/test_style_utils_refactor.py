from src.reports.style_utils import build_plot_card, embed_image, render_uw_css


def test_embed_image(tmp_path):
    img_path = tmp_path / "test.png"
    pixel_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00"
        b"\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n"
        b"\x02\xb1\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path.write_bytes(pixel_data)

    img_html = embed_image(img_path, alt="Test Alt")
    assert "data:image/png;base64," in img_html
    assert "alt='Test Alt'" in img_html

    # Missing file
    missing_html = embed_image(tmp_path / "missing.png", alt="Missing")
    assert "Plot unavailable" in missing_html


def test_build_plot_card():
    img_html = "<img src='foo' />"
    card = build_plot_card(img_html, "Test Caption")
    assert "plot-card" in card
    assert img_html in card
    assert "<figcaption>Test Caption</figcaption>" in card


def test_css_left_arrow():
    css = render_uw_css()
    assert "justify-content: flex-start" in css
    assert "padding: 0 0.5rem 0 0" in css  # arrow padding
