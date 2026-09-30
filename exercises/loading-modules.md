## Modules

One of the basic features of almost every HPC system is the existence of _modules_. Modules are in essence self-contained software installations, usually with multiple versions of every software. Any given software will depend on various libraries and other pieces of software, which leads to the concept of _toolchains_.

For example, a basic library in Linux installations is GCC, the GNU compiler collection, which contains compilers for languages like C and Fortran. A given version of a particular library might require a particular minimum version of GCC, for example `12.2.0`. It is generally best to use the same version of GCC to compile different libraries, and therefore, there will be a particular set of versions which depend on `GCC/12.2.0` We can find the versions of GCC available by typing `module load GCC` into the terminal, and hitting the Tab key twice:

```bash
$ module load GCC
GCC/            GCC/13.3.0      GCC/15.2.0      GCCcore/13.2.0  GCCcore/14.3.0  
GCC/12.3.0      GCC/14.2.0      GCCcore/        GCCcore/13.3.0  GCCcore/15.2.0  
GCC/13.2.0      GCC/14.3.0      GCCcore/12.3.0  GCCcore/14.2.0  
```

We don't need to load these modules explicitly, but `GCCcore`, a subset of `GCC`, defines the starting point of a _toolchain_. If we look at the available versions of `Python 3` by typing `Python/3` and hitting the tab key twice, we obtain

```bash
Python/3.11.3-GCCcore-12.3.0  Python/3.12.3-GCCcore-13.3.0  Python/3.13.5-GCCcore-14.3.0  
Python/3.11.5-GCCcore-13.2.0  Python/3.13.1-GCCcore-14.2.0  Python/3.14.2-GCCcore-15.2.0  
```

We therefore see that if we have an application that is limited to `Python 3.12`, we are automatically limited to the toolchains `GCCcore-13.3.0` and `GCCcore-13.3.0`. We must therefore make sure that any other packages that we want to use are either available from the module system, or can be loaded. For example, if we want to use `SciPy-bundle` which many commonly used scientific Python modules, we might start with just listing the versions available:

```bash
$ module load SciPy-bundle/202
SciPy-bundle/                     SciPy-bundle/2024.05-gfbf-2024a   SciPy-bundle/2026.05-gfbf-2026.1
SciPy-bundle/2023.07-gfbf-2023a   SciPy-bundle/2025.06-gfbf-2025a   
SciPy-bundle/2023.11-gfbf-2023b   SciPy-bundle/2025.07-gfbf-2025b   
```

There are two parallel toolchains here - `intel` and `foss`/`gfbf`. `GCCcore` is part of the `foss`/`gfbf` family, so the version we are looking for is somewhere in here. An easy approach to find the right version is to first load the `Python` version we want, and then simply try to load versions until we get the right one. If we load an incorrect version, then `Lmod`, which is what makes the `module` system work, will throw an error:

```bash
$ module load SciPy-bundle/2026.05-gfbf-2026.1
Lmod has detected the following error:  Attempted to load GCCcore/15.2.0 but GCCcore/13.3.0 was
already loaded.

For more info, see
https://www.c3se.chalmers.se/documentation/module_system/modules/#finding-compatible-software

...

$ module load SciPy-bundle/2024.05-gfbf-2024a
$ 
```

We can confirm by checking our loaded modules:

```bash
$ module list

Currently Loaded Modules:
  1) GCCcore/13.3.0                  15) FlexiBLAS/3.4.4-GCC-13.3.0
  2) zlib/1.3.1-GCCcore-13.3.0       16) FFTW/3.3.10-GCC-13.3.0
  3) binutils/2.42-GCCcore-13.3.0    17) gfbf/2024a
  4) bzip2/1.0.8-GCCcore-13.3.0      18) cffi/1.16.0-GCCcore-13.3.0
  5) ncurses/6.5-GCCcore-13.3.0      19) cryptography/42.0.8-GCCcore-13.3.0
  6) libreadline/8.2-GCCcore-13.3.0  20) virtualenv/20.26.2-GCCcore-13.3.0
  7) Tcl/8.6.14-GCCcore-13.3.0       21) Python-bundle-PyPI/2024.06-GCCcore-13.3.0
  8) SQLite/3.45.3-GCCcore-13.3.0    22) gzip/1.13-GCCcore-13.3.0
  9) XZ/5.4.5-GCCcore-13.3.0         23) lz4/1.9.4-GCCcore-13.3.0
 10) libffi/3.4.5-GCCcore-13.3.0     24) zstd/1.5.6-GCCcore-13.3.0
 11) OpenSSL/3                       25) ICU/75.1-GCCcore-13.3.0
 12) Python/3.12.3-GCCcore-13.3.0    26) Boost/1.85.0-GCC-13.3.0
 13) GCC/13.3.0                      27) pybind11/2.12.0-GCC-13.3.0
 14) AOCL-BLAS/5.0-GCC-13.3.0        28) SciPy-bundle/2024.05-gfbf-2024a
```

Now we know in the future that `GCCcore-13.3.0` corresponds to `foss/gfbf/gompi-2024a`.
In case it is not possible to find compatible versions for your module, a container is often a better choice.

The module system can be a bit tedious to use, but it typically comes down to finding which toolchain and pick what you need.
Apart from Python, it also has a ton of other optimized software.
