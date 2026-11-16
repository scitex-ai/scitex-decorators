---
description: |
  [TOPIC] scitex-decorators Installation
  [DETAILS] pip install scitex-decorators; the single [all] extra adds joblib, pandas, torch and xarray for the decorators that need those libraries.
tags: [scitex-decorators-installation]
---

# Installation

## Standard

```bash
pip install scitex-decorators
```

Pure-Python core. Type-conversion / batching / lifecycle decorators all
work out-of-the-box; the type-conversion decorators that target a
specific library (`@torch_fn`, `@pandas_fn`, `@xarray_fn`) only become
useful once that library is installed.

## Optional capabilities: `[all]`

`all` is the only extra. It adds joblib (`@cache_disk`,
`@cache_disk_async`), pandas, torch and xarray (`@pandas_fn`, `@torch_fn`,
`@xarray_fn`, `@signal_fn`, `@batch_torch_fn`, ...). A decorator whose
library is missing raises an `ImportError` naming this command:

```bash
pip install 'scitex-decorators[all]'
```

`@torch_fn` and friends place tensors on CUDA only when the installed torch
build has kernels for the present GPU; otherwise they warn once and use the
CPU.

## Verify

```bash
python -c "import scitex_decorators; print(scitex_decorators.__version__)"
python -c "from scitex_decorators import numpy_fn, batch_fn, deprecated, timeout; print('ok')"
```

## Editable install (development)

```bash
git clone https://github.com/ywatanabe1989/scitex-decorators
cd scitex-decorators
pip install -e '.[dev]'
```
