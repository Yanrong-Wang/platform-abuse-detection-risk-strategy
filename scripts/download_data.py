"""Download and verify the public YelpChi source; raw data stays out of Git."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip"
ZIP_SHA256 = "3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3"
MAT_SHA256 = "fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    directory = ROOT / "data/raw"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "YelpChi.zip"
    source = directory / "YelpChi.mat"
    if not archive.exists() or digest(archive) != ZIP_SHA256:
        urlretrieve(URL, archive)
    if digest(archive) != ZIP_SHA256:
        raise ValueError("Downloaded archive checksum mismatch")
    with ZipFile(archive) as z:
        if z.namelist() != ["YelpChi.mat"]:
            raise ValueError("Unexpected archive contents")
        z.extract("YelpChi.mat", directory)
    if digest(source) != MAT_SHA256:
        raise ValueError("Extracted dataset checksum mismatch")
    print(f"Verified: {source}")


if __name__ == "__main__":
    main()
