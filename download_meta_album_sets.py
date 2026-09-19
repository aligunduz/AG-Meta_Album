"""Download the official Meta-Album Mini Set-1 and Set-2 datasets from OpenML."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import openml


DATASETS = {
    "set1": {
        "DOG": 44298,
        "INS_2": 44292,
        "PLT_NET": 44293,
        "MED_LF": 44299,
        "PNU": 44297,
        "RSICB": 44300,
        "APL": 44295,
        "TEX_DTD": 44294,
        "ACT_40": 44291,
        "MD_5_BIS": 44296,
    },
    "set2": {
        "AWA": 44305,
        "INS": 44306,
        "FNG": 44302,
        "PLT_DOC": 44303,
        "PRT": 44308,
        "RSD": 44307,
        "BTS": 44309,
        "TEX_ALOT": 44304,
        "ACT_410": 44301,
        "MD_6": 44310,
    },
}


def validate_dataset(path: Path) -> tuple[int, int]:
    info_path = path / "info.json"
    labels_path = path / "labels.csv"
    images_path = path / "images"
    if not (info_path.is_file() and labels_path.is_file() and images_path.is_dir()):
        raise RuntimeError(f"Eksik Meta-Album dosyaları: {path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    description = info.get("dataset_description", info.get("Description", ""))
    if "MINI" not in str(description).upper():
        raise RuntimeError(f"Mini sürüm doğrulanamadı: {path}")

    with labels_path.open(encoding="utf-8-sig", newline="") as labels_file:
        label_count = sum(1 for _ in csv.DictReader(labels_file))
    image_count = sum(1 for item in images_path.iterdir() if item.is_file())
    if label_count != image_count:
        raise RuntimeError(
            f"Etiket/görüntü sayısı uyuşmuyor: {path} ({label_count}/{image_count})"
        )
    class_count = info.get("total_categories", info.get("Total Number of Classes"))
    if class_count is None:
        raise RuntimeError(f"Sınıf sayısı bulunamadı: {path}")
    if image_count != int(class_count) * 40:
        raise RuntimeError(
            f"Mini örnek sayısı beklenenden farklı: {path} "
            f"({image_count} != {class_count} x 40)"
        )
    return image_count, int(class_count)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    output_root = project_root / "meta_album_sets"
    cache_root = project_root / "meta_album_openml_cache"
    openml.config.set_root_cache_directory(str(cache_root))
    openml_cache = Path(openml.config.get_cache_directory())

    total_images = 0
    for set_name, datasets in DATASETS.items():
        for dataset_name, openml_id in datasets.items():
            destination = output_root / set_name / dataset_name
            if destination.exists():
                image_count, class_count = validate_dataset(destination)
                total_images += image_count
                print(
                    f"[ATLA] {set_name}/{dataset_name}: "
                    f"{image_count} görüntü, {class_count} sınıf",
                    flush=True,
                )
                continue

            print(
                f"[İNDİR] {set_name}/{dataset_name} (OpenML {openml_id})",
                flush=True,
            )
            openml.datasets.get_dataset(
                openml_id,
                download_data=True,
                download_all_files=True,
            )
            dataset_cache = openml_cache / "datasets" / str(openml_id)
            candidates = [
                item
                for item in dataset_cache.iterdir()
                if item.is_dir() and item.name.lower().endswith("_mini")
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"OpenML Mini klasörü bulunamadı: {dataset_cache} "
                    f"(adaylar: {[item.name for item in candidates]})"
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidates[0], destination)
            image_count, class_count = validate_dataset(destination)
            total_images += image_count
            print(
                f"[TAMAM] {set_name}/{dataset_name}: "
                f"{image_count} görüntü, {class_count} sınıf",
                flush=True,
            )

    print(f"[BİTTİ] Toplam {total_images} görüntü doğrulandı.", flush=True)


if __name__ == "__main__":
    main()
