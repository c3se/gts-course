## Connecting
### SSH

The standard way to connect to a computing cluster is to use the Secure Shell (SSH) protocol.

If you are using a computer with Linux or Mac OS X, and you are on Chalmers campus or connected to the Chalmers network via VPN, connecting to Vera via SSH is as simple as opening a terminal and typing

```bash
ssh CID@vera1.c3se.chalmers.se
```

Replace `CID` with your Chalmers ID. When prompted, enter your password.

For Windows, you might need to enable [OpenSSH](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse), but after that, you should be able to open a `PowerShell` or `cmd` instance and connect using the same command. On older Windows systems, you might need to install additional software, such as [PuTTy](https://www.chiark.greenend.org.uk/~sgtatham/putty/).

### Open OnDemand Portal

You can also connect to the cluster using the Open OnDemand Portal. For Alvis, the portal is located at [https://portal.c3se.chalmers.se](https://portal.c3se.chalmers.se); for Vera, it is at [https://vera.c3se.chalmers.se](https://vera.c3se.chalmers.se). As with SSH connection, you need to be at campus or use a VPN to connect. Simply follow the on-screen instructions to connect to the portal, and click "Interactive Apps" in the top bar. You can then launch a desktop session on a compute node, or an application such as a Jupyter Notebook.

## Transferring files

You can transfer files from your computer and vice versa using a variety of utilities. For Unix-like systems, the most common ones are `scp` and `rsync`. For Windows, `scp` is implemented natively since Windows 10. To use `scp` to transfer files from a Unix-like system to Vera, simply type

```bash
scp -r /path/to/your/folder CID@vera1.c3se.chalmers.se:/cephyr/users/CID/Vera/
```

Note that the path `/cephyr/users/CID/Vera` is your _home_ directory and can be replaced with the shorthand `~`. This will copy your folder and its contents to your home directory on Vera. On Windows, the syntax is similar:

```bash
scp -r "C:\path\to\your\folder" CID@vera1.c3se.chalmers.se:/cephyr/users/CID/Vera/
```

If you are only copying a single file, you can omit "-r". To copy files from Vera to your local computer, the easiest approach is to run `scp` from your local computer and simply exchange the places of the local and remote directory:


```bash
scp -r CID@vera1.c3se.chalmers.se:/cephyr/users/CID/Vera/ /path/to/your/folder 
```

Note that on older Windows versions, you may need to install additional utilities, such as WinSCP.

If you are on a Unix-like system, and you want to copy a large number of files in a more robust fashion, with the possibility to resume transfer and avoiding the copying of duplicates, [rsync](https://linux.die.net/man/1/rsync) may be a better option.

## SSH keys

To connect in a more secure way, and to avoid having to type your password each time you connect, it is a good idea to set up an SSH public-private key pair and put the public key on the remote.

```bash
ssh-keygen -t rsa
```

You will be prompted to enter a password for the SSH key; you are strongly recommended to do this in order to keep your access secure. There are a few ways to copy your public key to the remote server, which depend a bit on your system. A way that should generally work is to scp the key to your home directory, log in, and append it to the file `.ssh/authorized_keys`. Appending it without overwriting the `authorized_keys` allows you to create additional keys for other computers you may wish to log in from.

```bash
scp .ssh/id_rsa.pub CID@vera1.c3se.chalmers.se:/cephyr/users/CID/Vera/
ssh CID@vera1.c3se.chalmers.se
echo id_rsa.pub >> .ssh/authorized_keys  # Append public key to authorized_keys.
chmod go-rwx .ssh/ .ssh/authorized_keys  # Ensure permissions of key and directory are correct.
rm id_rsa.pub  # Remove public key file from Vera
```

Never share your private key, called simply `id_rsa`, with anyone.

## SSH configuration

We can set up an ssh configuration to simplify connecting to the cluster. Create the file `.ssh/config` and put into it:

```bash
Host vera1
    HostName vera1.c3se.chalmers.se
    IdentityFile ~/.ssh/id_rsa
    User CID
```

Now you can simply type `ssh vera1` to connect to the cluster. If you want to, you can add a persistent connection to simplify automated or other frequent connections:
