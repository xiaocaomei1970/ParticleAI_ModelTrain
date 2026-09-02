"""Particle flow-field training package.

Keep package import lightweight so dataset and manifest tools can run before
the GPU training dependencies are installed. Import training classes from their
concrete modules when needed, for example ``from model_flow.config import Config``.
"""

__all__: list[str] = []
