from pathlib import Path
import cv2
import numpy as np

def main() -> None:
    in_path = Path("pc/data/input/dataset-cover.png")

    # Output paths
    out_dir = Path("pc/data/prepped")
    out_png = out_dir / "dataset-cover_128x128.png"
    out_raw = out_dir / "dataset-cover_128x128.raw"

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image as grayscale (1 channel)
    img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {in_path}")

    print("Loaded image:")
    print(" - dtype:", img.dtype)
    print(" - shape:", img.shape)  # (H, W)
    # Resize to 128x128
    img_128 = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    print("Resized image:")
    print(" - dtype:", img_128.dtype)
    print(" - shape:", img_128.shape)
    # Save resized PNG for visual check
    ok = cv2.imwrite(str(out_png), img_128)
    if not ok:
        raise ValueError(f"Failed to write PNG: {out_png}")

    # Save RAW bytes (row-major, 1 byte per pixel)
    out_raw.write_bytes(img_128.tobytes())

    print("Saved files OK.")

    print("OK: script skeleton created")
    print(f"Input:  {in_path}")
    print(f"Output: {out_png}")
    print(f"Output: {out_raw}")


if __name__ == "__main__":
    main()
