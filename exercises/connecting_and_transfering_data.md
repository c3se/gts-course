## Connecting
### SSH

The standard way to connect to a computing cluster is to use the Secure Shell (SSH) protocol.

If you are using a computer with Linux or Mac OS X, and you are on Chalmers campus or connected to the Chalmers network via VPN, connecting to Vera via SSH is as simple as opening a terminal and typing

```bash
ssh CID@vera1.c3se.chalmers.se
```

When prompted, enter your password.

For Windows, you might need to enable [OpenSSH](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse), but after that, you should be able to open a `PowerShell` or `cmd` instance and connect using the same command. On older Windows systems, you might need to install additional software, such as [PuTTy](https://www.chiark.greenend.org.uk/~sgtatham/putty/).

### Open OnDemand Portal

You can also connect to the cluster using the Open OnDemand Portal. For Alvis, the portal is located at [https://portal.c3se.chalmers.se](https://portal.c3se.chalmers.se); for Vera, it is at [https://vera.c3se.chalmers.se](https://vera.c3se.chalmers.se). As with SSH connection, you need to be at campus or use a VPN to connect. Simply follow the on-screen instructions to connect to the portal, and click "Interactive Apps" in the top bar. You can then launch a desktop session on a compute node, or an application such as a Jupyter Notebook.

## Transferring files

You can transfer files from your computer and vice versa using a variety of utilities. For Unix-like systems, the most common ones are `scp` and `rsync`. For Windows, `scp` is implemented natively since Windows 10. To use `scp` to transfer files from a Unix-like system to Vera, simply type

```bash
scp -r /path/to/your/folder CID@vera1.c3se.chalmers.se:/cephyr/users/CID/Vera/
```

This will copy your folder and its contents to your home directory on Vera. On Windows, the syntax is similar:

```bash
scp -r "C:\path\to\your\folder" CID@vera1.c3se.chalmers.se:/cephyr/users/CID/Vera/
```

If you are only copying a single file, you can omit "-r". To copy files from Vera to your local computer, the easiest approach is to run `scp` from your local computer and simply exchange the places of the local and remote directory:


```bash
scp -r CID@vera1.c3se.chalmers.se:/cephyr/users/CID/Vera/ /path/to/your/folder 
```

Note that on older Windows versions, you may need to install additional utilities, such as WinSCP.

If you are on a Unix-like system, and you want to copy a large number of files in a more robust fashion, with the possibility to resume transfer and avoiding the copying of duplicates, [rsync](https://linux.die.net/man/1/rsync) may be a better option.
