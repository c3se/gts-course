## Containers

A container is a self-contained filesystem which functions a lot like a separate operating system. It can have its own software and libraries installed, and can be based on a different linux distribution than the system it is run in. This allows both a portable environment that allows you to instantly recreate your configuration on a different system, as well as the ability to install software which may have very specific requirements. On `vera`, you can use `apptainer` to create containers.

We can create a basic container based on a `docker` image, which is a different type of container. There are many `docker` images available in various repositories, and they are often a good starting point for building a container. We will use `python:slim-bookworm`, a Python installation in a Debian 12 (codenamed Bookworm) operating system. Debian is a commonly used Linux distribution which is also the basis for many other distributions, such as Ubuntu. It is called `slim` because it does not come with a lot of common utilities, such as `git`, in order to keep the file size down. If you need to use such utilities, you might want to use a different base container. 

We will install `numpy` in the container. Create a file called `python-bookworm.def` (you can use the command `touch apptainer.def`) and fill it with the following text (you can use a terminal text editor like `nano` or `vim`):

```apptainer
bootstrap: docker
from: python:slim-bookworm

%post
    python3 -m pip install numpy
```

To build the container, run `apptainer build python-bookworm.sif python-bookworm.def`.
Once the container finished building, run the following:

```bash
$ apptainer run python-bookworm.sif python -c "import numpy as np; print(np.arange(20))"`
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
```

Containers are often a good solution for installing obscure or customized software on your system, as the entire container is compressed into a single file, rather than many small files.
