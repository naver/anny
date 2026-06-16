# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import pathlib
import os
import logging

from anny.typing import PathLike

ANNY_ROOT_DIR = pathlib.Path(__file__).resolve().parent

# Define the default cache directory as a fixed, absolute path (user-overridable)
DEFAULT_CACHE_PATH = pathlib.Path.home() / ".cache" / "anny"
_ANNY_CACHE_DIR = [pathlib.Path(os.getenv("ANNY_CACHE_DIR", str(DEFAULT_CACHE_PATH)))]

logger = logging.getLogger(__name__)

def get_anny_cache_path() -> pathlib.Path:
    """
    Get the path to the Anny cache directory.

    Returns:
        pathlib.Path: The path to the Anny cache directory.
    """
    return _ANNY_CACHE_DIR[0]

def set_anny_cache_path(path: PathLike):
    """
    Set the path to the Anny cache directory.

    Args:
        path (PathLike): The new path to the Anny cache directory.
    """
    global _ANNY_CACHE_DIR
    _ANNY_CACHE_DIR[0] = pathlib.Path(path)

def get_anny2smplx_data_path() -> pathlib.Path:
    """
    Get the path to the Anny2SMPLX data file.

    Returns:
        pathlib.Path: The path to the Anny2SMPLX data file.
    """
    path = get_anny_cache_path() / "noncommercial/anny2smplx.pth"
    if not path.exists():
            download_noncommercial_data()
    return path

def get_anny2smpl_data_path() -> pathlib.Path:
    """
    Get the path to the Anny2SMPL data file.

    Returns:
        pathlib.Path: The path to the Anny2SMPL data file.
    """
    path = get_anny_cache_path() / "noncommercial/anny2smpl.pth"
    if not path.exists():
        download_noncommercial_data()
    return path

def download_noncommercial_data():
    cache_dir = get_anny_cache_path()
    noncommercial_data_url = "https://download.europe.naverlabs.com/humans/Anny/noncommercial.zip"
    dest_path = cache_dir / "noncommercial"

    logger.info("Downloading non-commercial data...")
    dest_path.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "noncommercial.zip"

    # Download the file
    import requests
    response = requests.get(noncommercial_data_url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)

    # Unzip the file
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_path)

    # Show the license file
    license_file = dest_path / "LICENSE.txt"
    if license_file.exists():
        logger.info("License Information:")
        logger.info("---------------------")
        with open(license_file, 'r') as f:
            logger.info(f.read())
    else:
        logger.info("LICENSE.txt file not found.")
    logger.info("-------------------")

    # Show the notice file
    notice_file = dest_path / "NOTICE.txt"
    if notice_file.exists():
        logger.info("-------------------")
        with open(notice_file, 'r') as f:
            logger.info(f.read())
    else:
        logger.info("NOTICE.txt file not found.")
    logger.info("-------------------")

    # Clean up the zip file
    os.remove(zip_path)
