#!/usr/bin/env python3
"""
DreamBooth Data Preprocessing: Interactive Crop Review Interface
Paths:
  Input : /home/qianqz/DreamBooth/origin-data
  Output: /home/qianqz/DreamBooth/resize-data
Controls:
  Arrow keys           Translate the crop window by STEP pixels.
  Left mouse button    Reposition the crop center at the selected point.
  Y / Enter            Confirm and save the current crop.
  N                    Skip the current image without saving.
  Q                    Exit the application.
"""

from pathlib import Path
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.widgets as widgets

# ================================================================
# Configuration
# ================================================================
INPUT_DIR   = Path("D:\\HCP\\67\\DreamBooth\\origin-data")
OUTPUT_DIR  = Path("D:\\HCP\\67\\DreamBooth\\resize-data")

TARGET_SIZE = 512          # Output resolution for square crops.
FILE_PREFIX = "instance"   # Prefix used for exported filenames.
OUTPUT_EXT  = ".jpg"       # Output image format.
STEP        = 20           # Pixel displacement for each arrow-key action.

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".heic"}
# ================================================================


def fix_exif_rotation(img: Image.Image) -> Image.Image:
    """Correct image orientation based on EXIF rotation metadata."""
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orient_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orient_key and orient_key in exif:
            rotation_map = {3: 180, 6: 270, 8: 90}
            angle = rotation_map.get(exif[orient_key])
            if angle:
                img = img.rotate(angle, expand=True)
    except Exception:
        pass
    return img


def collect_images(input_dir: Path) -> list:
    """Collect all supported image files from a directory in filename order."""
    images = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ])
    return images


def apply_crop_and_resize(img: Image.Image, cx: int, cy: int) -> Image.Image:
    """
    Crop a centered square around (cx, cy), then resize it to TARGET_SIZE.
    The crop side length is min(image width, image height), with boundary clamping.
    """
    w, h = img.size
    half = min(w, h) // 2

    # Clamp the crop window so that it remains within the image boundaries.
    cx = max(half, min(cx, w - half))
    cy = max(half, min(cy, h - half))

    left   = cx - half
    top    = cy - half
    right  = cx + half
    bottom = cy + half

    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)


class CropReviewer:
    """
    Interactive crop review interface:
      - Left panel: source image with a red crop boundary.
      - Right panel: crop preview corresponding to the current selection.
    """

    def __init__(self, images: list):
        self.images      = images
        self.idx         = 0             # Index of the currently reviewed image.
        self.save_count  = 0             # Number of saved crops.
        self.state       = None          # Placeholder for user decision state.

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Load the first image and initialize the crop center at the image center.
        self.img_pil = None
        self.cx = self.cy = 0
        self._load_current()

        # -- Construct the matplotlib review window ---------------------------
        self.fig, (self.ax_orig, self.ax_crop) = plt.subplots(
            1, 2, figsize=(14, 7)
        )
        self.fig.canvas.manager.set_window_title("DreamBooth Crop Review")

        # User guidance displayed in the figure footer.
        self.fig.text(
            0.5, 0.01,
            "Arrow keys: move crop window | Left click: reposition | "
            "Y/Enter: save | N: skip | Q: quit",
            ha="center", fontsize=10, color="gray"
        )

        # Progress title.
        self.title = self.fig.suptitle("", fontsize=12)

        # Source image with crop boundary.
        self.im_orig = self.ax_orig.imshow(
            self.img_pil, aspect="auto"
        )
        self.ax_orig.set_title("Source Image (Red Box = Crop Region)")
        self.ax_orig.axis("off")
        self.rect = patches.Rectangle(
            (0, 0), 1, 1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        self.ax_orig.add_patch(self.rect)

        # Crop preview.
        self.im_crop = self.ax_crop.imshow(
            apply_crop_and_resize(self.img_pil, self.cx, self.cy),
            aspect="auto"
        )
        self.ax_crop.set_title(f"Crop Preview ({TARGET_SIZE}x{TARGET_SIZE})")
        self.ax_crop.axis("off")

        # Register interaction events.
        self.fig.canvas.mpl_connect("key_press_event",  self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._refresh()
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.show()

    # -- Data Operations ------------------------------------------------------

    def _load_current(self):
        """Load the current image and reset the crop center to the image center."""
        path = self.images[self.idx]
        self.img_pil = fix_exif_rotation(
            Image.open(path).convert("RGB")
        )
        w, h = self.img_pil.size
        self.cx = w // 2
        self.cy = h // 2

    def _clamp_center(self):
        """Constrain the crop center so the crop window stays within image bounds."""
        w, h = self.img_pil.size
        half = min(w, h) // 2
        self.cx = max(half, min(self.cx, w - half))
        self.cy = max(half, min(self.cy, h - half))

    # -- Interface Refresh ----------------------------------------------------

    def _refresh(self):
        """Refresh the source image display and the crop preview."""
        w, h = self.img_pil.size
        half = min(w, h) // 2

        # Update source-image display.
        self.im_orig.set_data(self.img_pil)
        self.im_orig.set_extent([0, w, h, 0])

        # Update red crop-boundary position.
        self.rect.set_xy((self.cx - half, self.cy - half))
        self.rect.set_width(2 * half)
        self.rect.set_height(2 * half)

        # Update crop preview.
        preview = apply_crop_and_resize(self.img_pil, self.cx, self.cy)
        self.im_crop.set_data(preview)
        self.im_crop.set_extent([0, TARGET_SIZE, TARGET_SIZE, 0])

        # Update progress title.
        path = self.images[self.idx]
        self.title.set_text(
            f"[{self.idx + 1}/{len(self.images)}]  {path.name}"
            f"  Original size {w}x{h}  |  Saved crops {self.save_count}"
        )

        # Fix axis limits to prevent automatic rescaling by imshow.
        self.ax_orig.set_xlim(0, w)
        self.ax_orig.set_ylim(h, 0)
        self.ax_crop.set_xlim(0, TARGET_SIZE)
        self.ax_crop.set_ylim(TARGET_SIZE, 0)

        self.fig.canvas.draw_idle()

    # -- Event Handling -------------------------------------------------------

    def _on_key(self, event):
        key = event.key

        # Arrow keys translate the crop window.
        if key == "left":
            self.cx -= STEP
        elif key == "right":
            self.cx += STEP
        elif key == "up":
            self.cy -= STEP
        elif key == "down":
            self.cy += STEP

        # Confirm and save.
        elif key in ("y", "enter"):
            self._save_current()
            self._next_or_exit()
            return

        # Skip the current image.
        elif key == "n":
            print(f"  [SKIP] {self.images[self.idx].name}")
            self._next_or_exit()
            return

        # Exit the review interface.
        elif key == "q":
            print(f"\nReview terminated. Total saved crops: {self.save_count}.")
            plt.close(self.fig)
            return

        self._clamp_center()
        self._refresh()

    def _on_click(self, event):
        """Move the crop center to the selected point in the source-image panel."""
        if event.inaxes != self.ax_orig:
            return
        if event.button != 1:  # Respond only to the left mouse button.
            return

        # In imshow coordinates, xdata and ydata directly correspond to image pixels.
        self.cx = int(event.xdata)
        self.cy = int(event.ydata)
        self._clamp_center()
        self._refresh()

    # -- Saving and Navigation ------------------------------------------------

    def _save_current(self):
        """Save the current crop to the output directory using standardized naming."""
        self.save_count += 1
        out_name = f"{FILE_PREFIX}_{self.save_count:03d}{OUTPUT_EXT}"
        out_path = OUTPUT_DIR / out_name

        result = apply_crop_and_resize(self.img_pil, self.cx, self.cy)
        result.save(out_path, format="JPEG", quality=95)

        src_name = self.images[self.idx].name
        w, h = self.img_pil.size
        print(
            f"  [SAVE] {src_name:<30} ({w}x{h}) "
            f"-> {out_name} ({TARGET_SIZE}x{TARGET_SIZE})"
        )

    def _next_or_exit(self):
        """Advance to the next image, or close the interface after completion."""
        self.idx += 1
        if self.idx >= len(self.images):
            print(f"\n✅ Image review completed. Total saved crops: {self.save_count}.")
            print(f"   Output directory: {OUTPUT_DIR}")
            plt.close(self.fig)
            return
        self._load_current()
        self._refresh()


# ================================================================
# Entry Point
# ================================================================
def main():
    if not INPUT_DIR.exists():
        print(f"[ERROR] Input directory does not exist: {INPUT_DIR}")
        return

    images = collect_images(INPUT_DIR)
    if not images:
        print(f"[ERROR] No supported image files were found in {INPUT_DIR}.")
        return

    print("=" * 55)
    print("  DreamBooth Interactive Crop Review")
    print("=" * 55)
    print(f"  Input directory : {INPUT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Images detected : {len(images)}")
    print(f"  Target size     : {TARGET_SIZE} x {TARGET_SIZE}")
    print(f"  Naming convention: {FILE_PREFIX}_001{OUTPUT_EXT}, ...")
    print("=" * 55)
    print()
    print("  Controls:")
    print("    Arrow keys   Move the crop window by 20 px per step")
    print("    Left click   Reposition the crop center in the source image")
    print("    Y / Enter    Confirm and save the current crop")
    print("    N            Skip the current image without saving")
    print("    Q            Exit the application")
    print()

    CropReviewer(images)


if __name__ == "__main__":
    main()
