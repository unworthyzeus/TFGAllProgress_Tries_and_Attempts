from __future__ import annotations

import sys

from evaluate_cgan import main


if __name__ == '__main__':
    if '--split' not in sys.argv:
        sys.argv.extend(['--split', 'val'])
    if '--save-heuristic-calibration' not in sys.argv:
        sys.argv.append('--save-heuristic-calibration')
    main()