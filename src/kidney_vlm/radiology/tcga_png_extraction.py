from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kidney_vlm.radiology.feature_registry import build_png_series_dir
from kidney_vlm.radiology.png_preprocessing import RadiologyPngRenderConfig, ensure_resized_png


@dataclass(frozen=True)
class SeriesPngExtractionResult:
    png_dir: str
    png_paths: tuple[str, ...]
    slice_count: int
    pngs_created: int


class TCGARadiologyPngExtractor:
    def __init__(
        self,
        *,
        root_dir: Path,
        raw_root: Path,
        png_root: Path,
        overwrite_pngs: bool,
        png_render_mode: str,
        prefer_dicom_voi: bool,
        apply_padding_mask: bool,
        png_resize: int | None,
    ) -> None:
        self.root_dir = root_dir.expanduser().resolve()
        self.raw_root = raw_root.expanduser().resolve()
        self.png_root = png_root.expanduser().resolve()
        self.overwrite_pngs = bool(overwrite_pngs)
        self.render_config = RadiologyPngRenderConfig(
            render_mode=str(png_render_mode).strip() or "ct_rgb_multiwindow",
            prefer_dicom_voi=bool(prefer_dicom_voi),
            apply_padding_mask=bool(apply_padding_mask),
            resize=None if png_resize is None else int(png_resize),
        )

        try:
            import pydicom
            try:
                from pydicom.pixels import apply_modality_lut, apply_voi_lut
            except ImportError:
                from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
        except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Radiology PNG persistence requires 'pydicom'. Install project dependencies first."
            ) from exc

        self.pydicom = pydicom
        self.apply_modality_lut = apply_modality_lut
        self.apply_voi_lut = apply_voi_lut

    def extract_series(
        self,
        *,
        series_dir: Path,
        usable_image_paths: Sequence[Path],
    ) -> SeriesPngExtractionResult:
        resolved_series_dir = series_dir.expanduser().resolve()
        usable_paths = [Path(path).expanduser().resolve() for path in usable_image_paths]
        png_dir = build_png_series_dir(
            root_dir=self.root_dir,
            raw_root=self.raw_root,
            png_root=self.png_root,
            series_dir=resolved_series_dir,
        )
        png_dir.mkdir(parents=True, exist_ok=True)

        pngs_created = 0
        resolved_png_paths: list[str] = []
        for dicom_path in usable_paths:
            png_path = png_dir / f"{dicom_path.stem}.png"
            resolved_png_paths.append(str(png_path.resolve()))
            if ensure_resized_png(
                dicom_path=dicom_path,
                png_path=png_path,
                overwrite_pngs=self.overwrite_pngs,
                pydicom=self.pydicom,
                apply_modality_lut=self.apply_modality_lut,
                apply_voi_lut=self.apply_voi_lut,
                render_config=self.render_config,
            ):
                pngs_created += 1

        return SeriesPngExtractionResult(
            png_dir=str(png_dir),
            png_paths=tuple(resolved_png_paths),
            slice_count=len(usable_paths),
            pngs_created=pngs_created,
        )
