runtime/
========
- python/   Portable Python (installed by install.bat).
- get-pip.py is NOT bundled in the repo. The previous copy was incomplete (truncated) and would cause syntax errors if run directly.
- install.bat downloads the full get-pip.py from https://bootstrap.pypa.io/get-pip.py when pip is missing, then runs it. For offline use, run install.bat once with network so pip is bootstrapped; afterwards start.bat works offline.
