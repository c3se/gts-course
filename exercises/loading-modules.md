## Modules

See also documentation at <https://www.c3se.chalmers.se/documentation/module_system/>

One of the basic features of almost every HPC system is the existence of _modules_. Modules are in essence self-contained software installations, usually with multiple versions of every software. Any given software will depend on various libraries and other pieces of software, which leads to the concept of _toolchains_.

For example, a basic component is GCC, the GNU compiler collection, which contains compilers for languages like C, C++ and Fortran. A given version of a particular library will be compiled with a particular compiler version, for example GCC `14.2.0`.
We typically update the software stack 1-2 times per year, currently on Vera:

| Release | GCC    | OpenMPI | Intel    | CUDA   | Python | (and much more...) |
| ------- | ------ | ------- | -------- | ------ | ------ | ------------------ |
| 2023a   | 12.3.0 | 4.1.5   | 2023.1.0 | 12.1.1 | 3.11.3 | ...                |
| 2023b   | 13.2.0 | 4.1.6   | 2023.2.1 | 12.4.0 | 3.11.5 | ...                |
| 2024a   | 13.3.0 | 5.0.3   | 2024.2.0 | 12.6.0 | 3.12.3 | ...                |
| 2025a   | 14.2.0 | 5.0.7   | 2025.1.1 | 12.8.0 | 3.13.1 | ...                |
| 2025b   | 14.3.0 | 5.0.8   | 2025.2.0 | 12.9.1 | 3.13.5 | ...                |
| 2026.1  | 15.2.0 | 5.0.10  | 2025.3.3 | 13.3.0 | 3.14.2 | ...                |

It is generally best to use the same version of GCC to compile different libraries, and libraries aren't cross-compatible with different versions; one needs to stick to one.
We can find the versions of GCC available by typing `module load GCC` into the terminal, and hitting the Tab key twice:

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
The module system can be a bit tedious to use starting out, but it typically comes down to finding which toolchain and pick what you need once per project.

Final notes:
* You'll also find a many commercial applications like MATLAB, Star-CCM+, ANSA, etc. with many version in the module tree to pick from.
* You can request software to be installed as modules, or just be updated for newer toolchains or versions: <https://supr.naiss.se/support>
* We collaborate via EasyBuild <https://github.com/easybuilders/easybuild-easyconfigs/>. If the software is there, we can typically easily install it.
* We will also have pre-built containers and datasets available via modules soon; Loading those modules records that they are being used and helps us to know what needs to be kept.
* The software we build in the module system has been optimized for our system, but if your workload accounts for 0.1% of the total cluster core-hours, it doesn't matter. 

