"""Shrink rendered diagrams in place: 256-colour palette PNG, visually lossless."""

from pathlib import Path

from PIL import Image

ASSET = Path(__file__).resolve().parent.parent

for path in sorted([*ASSET.glob("[0-9][0-9]-*.png"), *(ASSET / "slides").glob("s[0-9][0-9]-*.png")]):
    image = Image.open(path)
    if image.mode == "P":
        continue
    image = image.convert("RGB").quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG
    )
    image.save(path, optimize=True)
    print(f"{path.relative_to(ASSET)}  {path.stat().st_size // 1024} KB")
