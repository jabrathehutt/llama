import sys, types
# 1. Handle missing distutils for Python 3.13
d = types.ModuleType("distutils")
sys.modules["distutils"] = d
u = types.ModuleType("distutils.util")
sys.modules["distutils.util"] = u
setattr(d, "util", u)
u.strtobool = lambda v: 1 if str(v).lower() in ("y", "yes", "t", "true", "1") else 0

# 2. Handle missing GluonTS loss modules for the unpickler
m = types.ModuleType("gluonts.torch.modules.loss")
sys.modules["gluonts.torch.modules.loss"] = m
class ML: pass
m.NegativeLogLikelihood = ML
m.DistributionLoss = ML

# 3. Import the rest of the original script
