"""
Signal detector registry — imports all tiers to trigger auto-registration.

After importing this package, all 20 detectors are registered in
SignalDetectorRegistry via __init_subclass__.
"""

from alpha_lab.agents.signal_eng.detectors.tier1 import *  # noqa: F401, F403
from alpha_lab.agents.signal_eng.detectors.tier2 import *  # noqa: F401, F403
from alpha_lab.agents.signal_eng.detectors.tier3 import *  # noqa: F401, F403
