import os
from typing import Dict


def _download_gdrive(url: str, out_path: str) -> None:
    """
    Download a Google Drive file using gdown.
    """
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "Google Drive download requires `gdown`.\n"
            "Install it with: pip install gdown"
        ) from e

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    result = gdown.download(url, out_path, quiet=False, fuzzy=True)

    if not result or not os.path.exists(out_path):
        raise RuntimeError("Failed to download file from Google Drive.")


def _download_http(url: str, out_path: str, timeout: int = 60) -> None:
    """
    Download a normal HTTP(S) file.
    """
    try:
        import requests
    except ImportError as e:
        raise ImportError(
            "HTTP downloads require `requests`.\n"
            "Install it with: pip install requests"
        ) from e

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tmp_path = out_path + ".part"

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    os.replace(tmp_path, out_path)


def download_files(
    needed_files: Dict[str, str],
    data_directory: str,
    *,
    overwrite: bool = False,
) -> None:
    """
    Ensure required files exist locally and download missing ones.

    Parameters
    ----------
    needed_files:
        Dictionary mapping filename -> download URL
    data_directory:
        Local directory where files should exist
    overwrite:
        If True, re-download even if file exists (must be keyword argument, cannot be positional)
    """

    os.makedirs(data_directory, exist_ok=True)

    missing = {}

    for filename, url in needed_files.items():
        file_path = os.path.join(data_directory, filename)

        if overwrite or not os.path.exists(file_path):
            missing[filename] = url

    if not missing:
        print("All required files exist.")
        return

    print(f"Missing files: {list(missing.keys())}. Downloading...")

    for filename, url in missing.items():
        file_path = os.path.join(data_directory, filename)

        try:
            if "drive.google.com" in url:
                _download_gdrive(url, file_path)
            else:
                _download_http(url, file_path)

            print(f"Downloaded {filename}")

        except Exception as exc:
            print(f"Failed to download {filename}: {exc}")
            raise

    print("All missing file(s) downloaded.")

### Example Usage
'''
from modules.misc.file_downloader import download_files

needed_files = {
    "paranmt-300.model":
        "https://huggingface.co/fse/paranmt-300/resolve/main/paranmt-300.model"
}

download_files(needed_files = needed_files, data_directory="data/dataset", overwrite=False)
'''