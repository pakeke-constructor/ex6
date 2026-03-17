


import os as _os
import ex6 as _ex6_guard

# only load these agents when running from the ex6 project folder
if _os.getcwd() == _os.path.dirname(_os.path.abspath(_ex6_guard.__file__)):
    import _ex6.agents._ex6_agents


del _ex6_guard, _os

