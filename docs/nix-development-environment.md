# Nix development environment

`devenv.nix` supplies the system tools that Flare's preparation and native build
require on NixOS, and `uv` supplies the Python packages from `pyproject.toml` and
`uv.lock`. The shell pins nixpkgs at revision
`f5c082a40f7571c266e74e80ae2e68aadd8a9fc7` through `devenv.yaml` and
`devenv.lock`.

## Entering the shell

```sh
devenv shell
```

`direnv allow` loads the same shell through `.envrc`. The shell provides GCC
15.3.0, CMake 4.1.6, Ninja 1.13.2, protoc 34.1, rustc and cargo 1.95.0, uv
0.11.21, the CUDA 13.2.51 toolkit, and Python 3.12.14.

Python wheels link two libraries that NixOS keeps outside their default search
path: the driver's `libcuda.so.1` under `/run/opengl-driver/lib`, and
`libstdc++.so.6` from the C++ toolchain. The shell sets `LD_LIBRARY_PATH` to both,
to the CUDA toolkit's libraries, and to zlib. An import of `torch` fails with
`libstdc++.so.6: cannot open shared object file` outside this environment.

## Python environment

```sh
uv sync --locked --extra prepare --extra test \
  --no-install-package flashinfer-python --no-managed-python
```

`--no-managed-python` selects the shell's Python 3.12.14. An interpreter that uv
downloads itself cannot start on NixOS without `nix-ld`, which this environment
does not enable; the shell's interpreter satisfies the `>=3.12,<3.13` range and
the locked package set.

`--no-install-package flashinfer-python` skips building the Git revision pinned in
`pyproject.toml` when the environment uses a local checkout instead. Without a
local checkout, omit the flag.

Ignition applies three preparation patches to Torch and Triton before importing
them. Apply them before starting a tool that imports either package directly:

```sh
.venv/bin/python -m ignition.patches
```

The native CLI builds with the same shell:

```sh
cargo build --release --manifest-path flare/Cargo.toml
```

## Local FlashInfer checkout

An editable install replaces the pinned FlashInfer revision without changing the
checkout's Git state:

```sh
BUILD_NIXL_EP=0 BUILD_NCCL_EP=0 FLASHINFER_BUILD_NO_PIP=1 \
  uv pip install --python .venv/bin/python --no-deps -e ../flashinfer
```

The three flags select the checkout's own build switches: they skip the optional
NIXL and NCCL transport builds and the build hook's pip installs, none of which
supply the Python package or its b12x kernels. `uv sync` restores the Git
revision pinned in `uv.lock`.

## Verification

On the `architect` host (NVIDIA RTX PRO 6000 Blackwell, driver 615.78.08, all four
GPUs occupied by a serving Flare process), this environment ran:

- `uv sync --locked --extra prepare --extra test --no-install-package
  flashinfer-python --no-managed-python`: installed Torch 2.14.1+cu132, Triton
  3.8.0, torchvision 0.29.1+cu132, Transformers 5.21.0.dev0, and built
  flare-heliopause 0.1.0.
- `ignition.patches`: applied all three patches.
- A CPU smoke script with `CUDA_VISIBLE_DEVICES` empty: imported `torch`,
  `triton`, `flashinfer`, and `ignition`; ran a 256x256 matrix product;
  compiled a function through Inductor; and compiled a trivial kernel with
  `nvcc -arch=sm_120` to a cubin. `torch.cuda.is_available` returned `False`,
  as expected with no visible device.
- `torch.cuda.device_count()`: returned 4 with the driver visible. No CUDA
  context or device memory was created; the GPUs retained their serving
  allocations.
- `pytest tests/test_design_review_hook.py tests/test_scheduling_sim.py
  tests/test_scheduling_oracle.py tests/test_scheduling_utility.py`: 308 passed.
- `cargo build --release --manifest-path flare/Cargo.toml`: built `flare`.

The FlashInfer CPU tests in `tests/experimental/b12x/sequence/test_engram.py`
ran with `CUDA_VISIBLE_DEVICES` empty: 4 passed and 4 failed in the checkout's
in-progress Engram implementation, which constructs a test `caps` object without
the `max_tokens` field the disk cache reads and compares 8-entry scale blocks
against 256-entry columns. No GPU kernel was executed.
