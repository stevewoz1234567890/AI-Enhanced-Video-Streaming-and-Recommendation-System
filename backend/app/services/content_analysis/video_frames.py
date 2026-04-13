import shutil
import subprocess
import tempfile
from pathlib import Path


def extract_frame_paths(
    video_path: str,
    sample_fps: float,
    max_frames: int,
) -> tuple[str, list[str]]:
    """
    Decode frames with FFmpeg into a temp directory; returns (tmpdir, sorted jpg paths).
    Caller must shutil.rmtree(tmpdir) when done.
    """
    tmpdir = tempfile.mkdtemp(prefix="frames_")
    pattern = str(Path(tmpdir) / "f%04d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={sample_fps}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        pattern,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    paths = sorted(str(p) for p in Path(tmpdir).glob("f*.jpg"))
    if not paths:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise ValueError("No frames extracted; check that the file is a supported video and ffmpeg is installed.")
    return tmpdir, paths
