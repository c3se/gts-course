## Running jobs

The core functionality of high-performance computing clusters is the ability to schedule and dispatch _jobs_, through a scheduling system such as `slurm`. We will look at two different types of jobs - interactive jobs and batch jobs.

### Interactive jobs

When you first do `ssh vera1`, you end up on `vera1`, a _login node_. The login node is a point of access to the cluster from the outside world. However, it is extremely important that you do not run demanding computations on the login node, as many other users depend on it to access the cluster. In order to run jobs, you should use a _compute node_.

The easiest way to access a compute node is to use [the web portal](https://vera.c3se.chalmers.se). Once logging in, use the top bar and go to `Interactive jobs -> Desktop`. Using the drop-down menu, launch a simple 1-core job for 30 minutes. Once the job launches, you should be able to view an interactive desktop in your browser.

Open a terminal by clicking the black square in the lower middle tray on the desktop. How is the terminal different from when you logged into `vera1`? What is this compute node called?

While on a compute node, you can safely run demanding jobs. The interactive environment is especially suitable to testing out your job configuration and ensuring it works correctly; in some cases, it can be sufficient to carry out all of the work you need to do on the HPC system.

It is also possible to launch interactive jobs [via the srun command](https://www.c3se.chalmers.se/documentation/running_jobs/#running-interactive-jobs). However, this is generally not recommended, as this session will depend entirely on your connection to the login node; the web portal is more robust in this regard.

### Batch jobs

Batch jobs are non-interactive jobs; they require you to write a script specifying the steps which the job is to carry out. The advantages of batch jobs are twofold - first, in case the resources you request are not available, the batch job will simply sit in a queue until it can run.
Second, batch jobs are independent of user-facing infrastructure, such as the portal and login nodes, and therefore they are less likely to be interrupted by any issues with these systems.
Third, it is often the case that HPC problems require carrying out many separate, similar jobs with slightly different parameters. Batch jobs allow this type of dispatch procedure to be submitted and carrying out in a structured way.

A basic batch script has the following contents:

```bash
#!/bin/bash
#SBATCH -A PROJECT_NAME -p CLUSTER_NAME
#SBATCH -n NUMBER_OF_CPU_NODES
#SBATCH -t MAXIMUM_RUN_TIME
#SBATCH -o LOG_FILE

SCRIPT_GOES_HERE
...
```

For a simple test job, we might write into the file `test_sbatch.sh`:

```bash
#!/bin/bash
#SBATCH -A PROJECT_NAME -p vera
#SBATCH -n 1
#SBATCH -t 00:00:10
#SBATCH test_log.txt

echo Job successful!
```

To submit this job, we run

```bash
$ sbatch test_sbatch.sh 
```

To check the result, we look at the contents of `test_sbatch.txt`:

```bash
$ cat test_sbatch.txt
Job successful!
```

### Modules in batch jobs

If we want to use the module system in an interactive job, we need to specify all the module load operations inside the batch script. For example:


```bash
#!/bin/bash
#SBATCH -A PROJECT_NAME -p vera
#SBATCH -n 1
#SBATCH -t 00:00:10
#SBATCH -o test_log.txt

module load Python/3.10.8

python -c "print([i * i for i in range(10)])"
```

```bash
$ cat test_log.txt
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

If you want the log file to be uniquely named, so that subsequent submissions do not overwrite it, you can use `%j` in its name, which will be replaced with the unique job ID of your batch job.

### Job arrays

In order to submit many similar jobs, we can use job arrays. Job arrays allow us to submit multiple similar jobs, differing only by an index which is referred to by `${SLURM_ARRAY_TASK_ID}` in the script, and as `%a` in any header argument:

```bash
#!/bin/bash
#SBATCH -A PROJECT_NAME -p vera
#SBATCH -n 1
#SBATCH -t 00:00:10
#SBATCH -o test_log_%a.txt

module load Python/3.10.8

python -c "print(${SLURM_ARRAY_TASK_ID}, ': ', [i * i for i in range(${SLURM_ARRAY_TASK_ID}])"
```

```bash
$ cat test_log_%a.txt
10 :  [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
0 :  []
1 :  [0]
2 :  [0, 1]
3 :  [0, 1, 4]
4 :  [0, 1, 4, 9]
5 :  [0, 1, 4, 9, 16]
6 :  [0, 1, 4, 9, 16, 25]
7 :  [0, 1, 4, 9, 16, 25, 36]
8 :  [0, 1, 4, 9, 16, 25, 36, 49]
9 :  [0, 1, 4, 9, 16, 25, 36, 49, 64]
6 :  [0, 1, 4, 9, 16, 25]
7 :  [0, 1, 4, 9, 16, 25, 36]
8 :  [0, 1, 4, 9, 16, 25, 36, 49]
9 :  [0, 1, 4, 9, 16, 25, 36, 49, 64]
```

### Running jobs using GPU resources

In order to use GPU resources, we need to do two things: First, make sure we load a library that can use the GPU. Second, we need to request a node with a GPU. For this demonstration, we will use `CuPy`, a drop-in replacement for `numpy` that works with Nvidia GPU:s, and we will request an `A40` GPU using `gpus-per-node=A40:1`.


```bash
#!/bin/bash
#SBATCH -A PROJECT_NAME -p vera
#SBATCH -t 00:00:10
#SBATCH --gpus-per-node=A40:1
#SBATCH -o test_log_gpu.txt

module load Python/3.10.8 CuPy/12.1.0-foss-2022b-CUDA-12.0.0

python -c "import cupy as cp; array = cp.arange(1000); print(array.sum().get())"
```

Once the job is finished, you can check the output:

```bash
$ cat test_log_gpu.txt
499500
```

If your job does not execute immediately, check which GPU types are avaiable (`IDLE`) under `Total GPU usage`:

```bash
$ jobinfo
...
Total GPU usage:
TYPE    ALLOCATED IDLE OFFLINE TOTAL
A100            2   10       0    12
A40             4   12       0    16
T4              4    0       0     4
V100            0    8       0     8
...
```
