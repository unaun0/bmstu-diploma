from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.common.settings import IMAGE_EXTENSIONS
from src.common.utils import load_image_rgb
from src.training.utils import load_mask_binary


class FabricDefectDataset(Dataset):
    def __init__(self, root: Path, transform=None) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.transform = transform
        self.samples = self._collect()

    def _collect(self) -> list[tuple[Path, Path]]:
        samples = []
        for img in sorted(self.images_dir.glob("*")):
            mask = self._mask_path(img.stem)
            if mask:
                samples.append((img, mask))
        return samples

    def _mask_path(self, stem: str) -> Path | None:
        for ext in IMAGE_EXTENSIONS:
            path = self.masks_dir / f"{stem}{ext}"
            if path.exists():
                return path
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        img_path, mask_path = self.samples[index]
        image = load_image_rgb(img_path)
        mask = load_mask_binary(mask_path)

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        return {
            "image": image.float(),
            "mask": mask.float().unsqueeze(0),
            "name": img_path.stem,
        }
