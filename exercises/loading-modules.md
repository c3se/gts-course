## Modules

One of the basic features of almost every HPC system is the existence of _modules_. Modules are in essence self-contained software installations, usually with multiple versions of every software. Any given software will depend on various libraries and other pieces of software, which leads to the concept of _toolchains_.

For example, a basic library in Linux installations is GCC, the GNU compiler collection, which contains compilers for languages like C and Fortran. A given version of a particular library might require a particular minimum version of GCC, for example `12.2.0`. It is generally best to use the same version of GCC to compile different libraries, and therefore, there will be a particular set of versions which depend on `GCC/12.2.0` We can find the versions of GCC available by typing `module load GCC` into the terminal, and hitting the Tab key twice:

```bash
$ module load GCC
GCC             GCC/11.3.0      GCC/13.2.0      GCCcore/10.3.0  GCCcore/12.2.0  GCCcore/13.3.0
GCC/10.3.0      GCC/12.2.0      GCC/13.3.0      GCCcore/11.2.0  GCCcore/12.3.0  
GCC/11.2.0      GCC/12.3.0      GCCcore         GCCcore/11.3.0  GCCcore/13.2.0
```

We don't need to load these modules explicitly, but `GCCcore`, a subset of `GCC`, defines the starting point of a _toolchain_. If we look at the available versions of `Python 3` by typing `Python/3` and hitting the tab key twice, we obtain

```bash
Python/3.10.4-GCCcore-11.3.0       Python/3.11.3-GCCcore-12.3.0       Python/3.9.5-GCCcore-10.3.0-bare
Python/3.10.4-GCCcore-11.3.0-bare  Python/3.11.5-GCCcore-13.2.0       Python/3.9.6-GCCcore-11.2.0
Python/3.10.8-GCCcore-12.2.0       Python/3.12.3-GCCcore-13.3.0       Python/3.9.6-GCCcore-11.2.0-bare
Python/3.10.8-GCCcore-12.2.0-bare  Python/3.9.5-GCCcore-10.3.0
```

We therefore see that if we have an application that is limited to `Python 3.10`, we are automatically limited to the toolchains `GCCcore-11.3.0` and `GCCcore-12.2.0`. We must therefore make sure that any other packages that we want to use are either available from the module system, or can be loaded. For example, if we want to use `SciPy-bundle` which many commonly used scientific Python modules, we might start with just listing the versions available:

```bash
$ module load SciPy-bundle/202
SciPy-bundle/2021.05-foss-2021a   SciPy-bundle/2022.05-foss-2022a   SciPy-bundle/2023.07-iimkl-2023a
SciPy-bundle/2021.05-intel-2021a  SciPy-bundle/2022.05-intel-2022a  SciPy-bundle/2023.11-gfbf-2023b
SciPy-bundle/2021.10-foss-2021b   SciPy-bundle/2023.02-gfbf-2022b   SciPy-bundle/2024.05-gfbf-2024a
SciPy-bundle/2021.10-intel-2021b  SciPy-bundle/2023.07-gfbf-2023a
```

There are two parallel toolchains here - `intel` and `foss`/`gfbf`. `GCCcore` is part of the `foss`/`gfbf` family, so the version we are looking for is somewhere in here. An easy approach to find the right version is to first load the `Python` version we want, and then simply try to load versions until we get the right one. If we load an incorrect version, then `Lmod`, which is what makes the `module` system work, will throw an error:

```bash
$ module load SciPy-bundle/2023.07-gfbf-2023a 
Lmod has detected the following error:  Your site prevents the automatic swapping of modules
with same name. You must explicitly unload the loaded version of "GCC/12.2.0" before you can load the new
one. Use swap to do this:

   $ module swap GCC/12.2.0 GCC/12.3.0
...
$ module load SciPy-bundle/2023.02-gfbf-2022b
$
```

We can confirm by checking our loaded modules:

```bash
$ module list

Currently Loaded Modules:
  1) GCCcore/12.2.0                    18) FFTW/3.3.10-GCC-12.2.0
  2) zlib/1.2.12-GCCcore-12.2.0        19) gompi/2022b
  3) binutils/2.39-GCCcore-12.2.0      20) FFTW.MPI/3.3.10-gompi-2022b
  4) GCC/12.2.0                        21) ScaLAPACK/2.2.0-gompi-2022b-fb
  5) numactl/2.0.16-GCCcore-12.2.0     22) foss/2022b
  6) XZ/5.2.7-GCCcore-12.2.0           23) bzip2/1.0.8-GCCcore-12.2.0
  7) libxml2/2.10.3-GCCcore-12.2.0     24) ncurses/6.3-GCCcore-12.2.0
  8) libpciaccess/0.17-GCCcore-12.2.0  25) libreadline/8.2-GCCcore-12.2.0
  9) hwloc/2.8.0-GCCcore-12.2.0        26) Tcl/8.6.12-GCCcore-12.2.0
 10) OpenSSL/1.1                       27) SQLite/3.39.4-GCCcore-12.2.0
 11) libevent/2.1.12-GCCcore-12.2.0    28) GMP/6.2.1-GCCcore-12.2.0
 12) UCX/1.13.1-GCCcore-12.2.0         29) libffi/3.4.4-GCCcore-12.2.0
 13) PMIx/4.2.2-GCCcore-12.2.0         30) Python/3.10.8-GCCcore-12.2.0
 14) UCC/1.1.0-GCCcore-12.2.0          31) gfbf/2022b
 15) OpenMPI/4.1.4-GCC-12.2.0          32) pybind11/2.10.3-GCCcore-12.2.0
 16) OpenBLAS/0.3.21-GCC-12.2.0        33) SciPy-bundle/2023.02-gfbf-2022b
 17) FlexiBLAS/3.2.1-GCC-12.2.0
```

Now we know in the future that `GCCcore-12.2.0` corresponds to `foss/gfbf/gompi-2022b`. In case it is not possible to find compatible versions for your module, a container is often a better choice.

Thus, the module system can be a bit tedious to use, as it requires some trial and error to find the correct versinos combinations, but when it works it is the most convenient way to use the cluster, since module are tested as they are built, and most of the time will "just work".
