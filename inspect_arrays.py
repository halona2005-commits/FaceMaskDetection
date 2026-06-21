import numpy as np
from pathlib import Path
p = Path(__file__).resolve().parent / 'data'
files = ['X_train.npy','X_test.npy','y_train.npy','y_test.npy']
for f in files:
    path = p / f
    if path.exists():
        arr = np.load(path)
        print(f, arr.shape, arr.dtype)
    else:
        print(f'MISSING: {path}')
