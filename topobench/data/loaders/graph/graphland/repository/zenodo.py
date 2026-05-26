import zipfile
from fnmatch import fnmatch
from io import BytesIO
from urllib.request import Request, urlopen


class ZenodoZip:
    """Tiny helper to download a ZIP to memory and return {name: bytes}.
    Only uses the Python standard library (urllib, zipfile).
    """

    def __init__(self, url: str, timeout: int = 60):
        self.url = url
        self.timeout = timeout

    def fetch(self, include=None):
        """Download and extract. `include` is an iterable of glob patterns
        (e.g., ["**/*.csv"]). If None, all files are returned.
        Returns: dict[str, bytes]
        """
        blob = self._download()
        return self._extract(blob, include)


    def _download(self) -> bytes:
        req = Request(self.url, headers={"User-Agent": "zenodo-zip-minimal/1.0"})
        with urlopen(req, timeout=self.timeout) as r:
            return r.read()


    def _extract(self, blob: bytes, include) -> dict:
        include = list(include) if include else ["*"]
        out = {}
        with zipfile.ZipFile(BytesIO(blob)) as zf:
            if zf.testzip() is not None:
                raise ValueError("Corrupted ZIP archive (testzip failed).")
            for info in zf.infolist():
                name = info.filename.replace("\\", "/")
                if name.endswith("/") or name.startswith("__MACOSX/"):
                    continue # skip dirs and macOS metadata
                if not any(fnmatch(name, pat) for pat in include):
                    continue
                out[name] = zf.read(info)
        return out
