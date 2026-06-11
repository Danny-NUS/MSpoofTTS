import os
import shutil
from pathlib import Path
from typing import Dict, List

# =========================================================
# CONFIG
# =========================================================

BASE_REMOTE = Path("/data2/minh_duc/neutts_eval")
OUT_DIR = Path("/data2/minh_duc/mos")

SCHEMES = [
    "orig",
    "ras_k50_win25",
    "eas",
    "rank_ras_hier",
    "rank_eas_hier",
]

DATASETS: Dict[str, List[str]] = {
    "Librispeech": [
        "61-70968-0000",
        "121-121726-0000",
        "237-126133-0000",
        "672-122797-0000",
        "5142-33396-0000",
        "7021-79730-0000",
        "7729-102255-0000",
        "8224-274381-0002",
        "8455-210777-0000",
        "8555-284447-0000",
    ],
    "Libritts": [
        "121_127105_000007_000002",
        "1089_134686_000032_000007",
        "1580_141083_000002_000002",
        "1995_1826_000005_000001",
        "2300_131720_000002_000001",
        "2961_961_000002_000000",
        "3570_5694_000005_000003",
        "3729_6852_000003_000004",
        "4077_13751_000006_000005",
        "4970_29093_000004_000000",
    ],
    "twistlist_test": [
        "twist_000000",
        "twist_000001",
        "twist_000006",
        "twist_000007",
        "twist_000013",
        "twist_000016",
        "twist_000021",
        "twist_000025",
    ],
}


# =========================================================
# FUNCTION
# =========================================================

def find_matching_wav(dataset: str, scheme: str, utt_id: str) -> Path | None:
    """
    Find file matching:
    {utt_id}__ref_*__k0.wav
    """
    search_dir = BASE_REMOTE / dataset.lower() / "syn" / scheme / "wav"

    if not search_dir.exists():
        return None

    pattern = f"{utt_id}__ref_"
    for file in search_dir.iterdir():
        if file.name.startswith(pattern) and file.name.endswith("__k0.wav"):
            return file

    return None


# =========================================================
# MAIN
# =========================================================

def main():
    print("Building MOS bundle...")

    for dataset, utt_list in DATASETS.items():
        print(f"\nDataset: {dataset}")

        for utt_id in utt_list:
            target_folder = OUT_DIR / dataset / utt_id
            target_folder.mkdir(parents=True, exist_ok=True)

            for scheme in SCHEMES:
                src = find_matching_wav(dataset, scheme, utt_id)

                if src is None:
                    print(f"Missing: {dataset} | {utt_id} | {scheme}")
                    continue

                dst = target_folder / f"{scheme}.wav"
                shutil.copy2(src, dst)
                print(f"Copied: {dataset}/{utt_id}/{scheme}.wav")

    print("\nFinished building MOS bundle.")
    print(f"Location: {OUT_DIR}")


if __name__ == "__main__":
    main()