"""Sandboxed execution + confinement.

Confinement primitives (`landlock`, `seccomp`, `egress`), the per-job orchestrator
(`runner`), the in-sandbox notebook entrypoint (`_run_entry`), and the capability
prober (`probe`). (Dataset predownload lives in the parent-level `data` module.)
"""

from .runner import Canceller, JobContext, SandboxResult, run_notebook, setup_job

__all__ = ["Canceller", "JobContext", "SandboxResult", "run_notebook", "setup_job"]
