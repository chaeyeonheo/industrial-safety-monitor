"""cv2.VideoWriter(fourcc='mp4v')로 만든 영상은 실제로는 FMP4(MPEG-4 Part 2)
코덱이라 브라우저 <video> 태그에서 재생이 안 된다(H.264만 널리 지원됨). 이미
만들어둔 mp4를 imageio-ffmpeg가 받아온 ffmpeg 바이너리로 H.264 재인코딩한다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def reencode(path: Path) -> None:
    tmp_path = path.with_suffix(".h264tmp.mp4")
    result = subprocess.run(
        [FFMPEG, "-y", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(tmp_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"실패: {path}\n{result.stderr[-500:]}")
        tmp_path.unlink(missing_ok=True)
        return
    tmp_path.replace(path)
    print(f"재인코딩됨: {path}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        reencode(Path(arg))
