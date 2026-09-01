"""Extract the published paper figures from the MDPI PDF at README resolution.

The crop rectangles are tied to Machines 2026, 14, 397 (PDF v2):
https://doi.org/10.3390/machines14040397
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class FigureCrop:
    number: int
    page: int
    rect: tuple[float, float, float, float]
    filename: str


FIGURES = (
    FigureCrop(1, 5, (166, 66, 560, 292), "figure-01-actor-architecture.png"),
    FigureCrop(2, 5, (166, 448, 560, 660), "figure-02-td3-framework.png"),
    FigureCrop(3, 6, (166, 67, 560, 255), "figure-03-twin-critics.png"),
    FigureCrop(4, 10, (166, 58, 560, 396), "figure-04-gnn-module.png"),
    FigureCrop(5, 12, (166, 465, 560, 740), "figure-05-transformer-encoder.png"),
    FigureCrop(6, 15, (166, 68, 380, 221), "figure-06-pybullet-environment.png"),
    FigureCrop(7, 16, (166, 108, 560, 310), "figure-07-training-task-metrics.png"),
    FigureCrop(8, 16, (166, 356, 560, 562), "figure-08-training-trajectory-metrics.png"),
    FigureCrop(9, 17, (166, 510, 560, 704), "figure-09-test-task-metrics.png"),
    FigureCrop(10, 18, (166, 67, 560, 254), "figure-10-test-trajectory-metrics.png"),
    FigureCrop(11, 19, (166, 146, 560, 413), "figure-11-stability-curves.png"),
    FigureCrop(12, 19, (166, 458, 560, 715), "figure-12-stability-distributions.png"),
    FigureCrop(13, 20, (166, 218, 560, 687), "figure-13-trajectory-tracking.png"),
)


def extract_figures(pdf_path: Path, output_dir: Path, zoom: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = pymupdf.Matrix(zoom, zoom)

    with pymupdf.open(pdf_path) as document:
        if document.page_count < max(figure.page for figure in FIGURES):
            raise ValueError(
                f"Expected at least {max(f.page for f in FIGURES)} pages, "
                f"found {document.page_count}."
            )

        for figure in FIGURES:
            page = document[figure.page - 1]
            clip = pymupdf.Rect(*figure.rect) & page.rect
            pixmap = page.get_pixmap(
                matrix=matrix,
                clip=clip,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            output_path = output_dir / figure.filename
            pixmap.save(output_path)
            print(
                f"Figure {figure.number:02d}: page {figure.page} -> "
                f"{output_path} ({pixmap.width}x{pixmap.height})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to machines-14-00397-v2.pdf")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/paper"),
        help="Destination directory (default: assets/paper)",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=3.0,
        help="Render scale; 3.0 produces GitHub-readable figures",
    )
    args = parser.parse_args()

    if args.zoom <= 0:
        parser.error("--zoom must be positive")
    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")

    extract_figures(args.pdf, args.output_dir, args.zoom)


if __name__ == "__main__":
    main()
