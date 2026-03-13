from __future__ import annotations

import sys

from evaluate import main


if __name__ == '__main__':
    if '--split' not in sys.argv:
        sys.argv.extend(['--split', 'val'])
    main()