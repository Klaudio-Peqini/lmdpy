
"""
lmdpy: Langevin Molecular Dynamics in Python

A flexible framework for:
- Single- and multi-particle Langevin simulations
- PDG-inspired coherent-domain potentials
- Generic classical potentials (harmonic, double-well, Lennard-Jones, Morse, etc.)
- Spatially and temporally varying solvent fields (friction, temperature, flow)
- HPC / HTCondor ensemble launches
- Publication-ready figure generation
- Simple web UI for teaching / interactive exploration

Core modules
------------
- lmdpy.potentials    : Potential classes (U, gradU)
- lmdpy.solvent       : Solvent/medium description (γ(x,t), T(x,t), v_flow(x,t))
- lmdpy.langevin_core : Generic Langevin integrator using potentials + solvent
- lmdpy.figures       : Matplotlib-based figure helpers
- lmdpy.hpc           : HPC / HTCondor helpers
- lmdpy.webui         : Flask-based web UI (see lmdpy/webui)

Existing stand-alone scripts from the original development are also provided
in the `examples/` directory (e.g. PDG-specific models).
"""

from . import potentials
from . import solvent
from . import langevin_core
from . import figures
from . import hpc

__all__ = [
    "potentials",
    "solvent",
    "langevin_core",
    "figures",
    "hpc",
]
