"""为中文 Benchmark 图表统一设置字体。

优先使用 MINIEP_PLOT_FONT，其次查找常见 Noto 中文字体；找不到时直接报错，
避免生成缺字方框的图。
"""

import os
from pathlib import Path

from matplotlib import font_manager, rcParams


def use_chinese_font() -> None:
    candidates = [
        os.environ.get("MINIEP_PLOT_FONT"),
        Path.home() / ".local/share/fonts/NotoSansSC-Regular.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        path = str(candidate)
        font_manager.fontManager.addfont(path)
        family = font_manager.FontProperties(fname=path).get_name()
        rcParams.update({"font.family": family, "svg.fonttype": "none"})
        return
    for family in ("Noto Sans SC", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        rcParams.update({"font.family": family, "svg.fonttype": "none"})
        return
    raise RuntimeError("Install Noto Sans SC or set MINIEP_PLOT_FONT to a Chinese font file")
