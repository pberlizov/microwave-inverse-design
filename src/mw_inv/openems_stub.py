"""Minimal openEMS / Octave geometry stub from ``CavityParams``.

Deprecated name — use ``mw_inv.openems_export`` for the full runnable model.
"""

from mw_inv.openems_export import export_scene_npz, generate_openems_script, write_openems_model

generate_openems_stub = generate_openems_script

__all__ = ["generate_openems_stub", "generate_openems_script", "write_openems_model", "export_scene_npz"]
