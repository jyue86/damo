from typing import Any, Mapping


class BaseLogger:
    def log_config(self, cfg: Any) -> None:
        pass

    def log(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str, name: str | None = None, type: str | None = None) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def finish(self) -> None:
        pass