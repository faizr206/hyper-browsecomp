#!/usr/bin/env python3
"""Rescore every ``.eval`` file under ``logs/valid results``."""

from rescore_eval_logs import LOG_ROOT, main


if __name__ == "__main__":
    raise SystemExit(main(LOG_ROOT / "valid results"))
