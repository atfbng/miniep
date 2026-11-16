"""验证启动配置，特意与生产默认配置分开。

除了安装 RouteEP 运行时环境，还设置 MINIEP_VALIDATION_MODE 和报告目录，
并改用 routeep_validation 插件，避免验证钩子混入正常推理。
"""

import os
from pathlib import Path
from mini_ep.integration.config import configure as configure_runtime


def configure(backend: str, output: str) -> None:
    if backend not in ("deepep_native", "deepep_exact"):
        raise ValueError(
            f"validation mode must be deepep_native/deepep_exact, got {backend!r}"
        )
    configure_runtime("native" if backend == "deepep_native" else "miniep")
    os.environ["MINIEP_VALIDATION_MODE"] = backend
    os.environ["VLLM_PLUGINS"] = "routeep_validation"
    Path(output).mkdir(parents=True, exist_ok=True)
    os.environ["MINIEP_REPORT_DIR"] = str(Path(output).resolve())
