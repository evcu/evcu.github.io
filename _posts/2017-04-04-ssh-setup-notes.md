---
layout: single
title: "SSH Setup Notes"
tags: [tutorial,basics,ssh,hpc]
category: notes
excerpt: "useful ssh setup commands for good"
---
If you are working remotely and connecting to some other machine via ssh all the time, sometimes it becomes too frustrating to enter the password every single time. Until you setup your key-pairs. I didn't know this until very recently. So in this post I will share very simple ssh setup basics. Lets say you want to do following connection:

> ssh test49@stampede.tacc.utexas.edu 

This post explains the following:
- How to create key-pair and share your public key with the remotes. Such that you can ssh without entering the password each time. This is especially useful when you are running distributed code. 
- How to create `config` file and create alias for ssh commands. So you can just call `ssh stampede` and you are connected to the machine! 


## Key-Pair Setup
This part is inspired by [this post](https://www.digitalocean.com/community/tutorials/how-to-set-up-ssh-keys--2). There are two steps to be made to share your public key with the host. 

- __Create the key pair__
    - Give location to be saved (Press `Enter` for default) 
    - Optional passphrase, which is asked if entered each time the private key is used (safer).
    - Keys reside (if not changed intentionally during creation) at `~/.ssh/id_rsa`(your private key) and `~/.ssh/id_rsa.pub`(your public key)

``` 
ssh-keygen -t rsa
```

- __Copy your public key to remote machine (using your password)__
    - You need to copy your public key to the remote hosts in order to be able to use your private key to connect.
    - Do this automatically by using.
``` 
ssh-copy-id test49@stampede.tacc.utexas.edu 
```
   You can also to this step manually like below
```
cat ~/.ssh/id_rsa.pub | ssh test49@stampede.tacc.utexas.edu  "mkdir -p ~/.ssh && cat >>  ~/.ssh/authorized_keys"
```
   After this step now you can connect to your server by running `ssh test49@stampede.tacc.utexas.edu` without entering any password(possibly passphrase for your private key)

## Alias Setup
You can save your known ssh connections to the `~/.ssh/config` file. Add following lines to this file. If not exist create it. 
```
Host stampede
  HostName stampede.tacc.utexas.edu
  User test49
```
   This will allow you to call `ssh stampede`. 

   If desired flags can be also added by appropriate field_names. Here are some example such names replacing the flags of the ssh program.

| ssh Flags | config line | 
| ---- | ----------------- |
|test49@stampede.tacc.utexas.edu | User test49|
|test49@stampede.tacc.utexas.edu | HostName stampede.tacc.utexas.edu |
| -p 22000 |  Port 22000 | 
| -i ~/.ssh/id_rsa | IdentityFile ~/.ssh/id_rsa |
| -L 8000:localhost:8888 |  LocalForward 8000 localhost:8888 |
| -X | ForwardX11 yes |

To find out all config options check out man file with `man ssh_config`. After entering some of the config options one can still use flags before the alias. An example being:

    ssh -f -N stampede

## SSH Tunneling
Sometimes connecting to a server requires two ssh connections if you are outside of the local network. This was the case when I worked off-campus and wanted to use NYU's clusters. Having two ssh connections makes local forwarding above a little more complicated. Without any tunneling one would need to arrange two local forwarding carefully. Another annoying thing when you work off-campus is that you need to make two ssh calls every time you want to open a new shell. Tunneling fixes both of the problems above. This part is inspired from the NYU's HPC [link](https://wikis.nyu.edu/display/NYUHPC/SSH+tunneling+overview). 

I will be using NYU HPC off-campus access as an example to explain tunneling. If you are off-campus you need to do an ssh call to `hpc.nyu.edu` first and than another one to `prince.hpc.nyu.edu`. Lets start with adding following lines to our `~/.ssh/config` file to define our tunnel. 

```
Host hpc2tunnel
  HostName hpc.nyu.edu
  ForwardX11 no
  LocalForward 8026 prince.hpc.nyu.edu:22
  LocalForward 8025 dumbo.hpc.nyu.edu:22
  User ue225
```

This defines the tunnel, which is nothing but an ssh connection with local forwarding connecting localhost:8026 to default ssh port(22) of `prince.hpc.nyu.edu` on `hpc.nyu.edu`(for a short history of how port 22 became the default port, read [here](https://www.ssh.com/ssh/port)). Other local forwarding definitions can be defined on the same Host definition. After this when `ssh hpc2tunnel` called, our localhost at port 8026 listens `prince.hpc.nyu.edu:22` on `hpc.nyu.edu` therefore we can ssh to our localhost at port 8026 to connect to the prince cluster with one call. Adding the definition below does this for us.
```
Host princeOffCampus
  HostName localhost
  Port 8026
  ForwardX11 yes
  User ue225
```

After these definitions all you need to do call `ssh hpc2tunnel` and __leave it open__. Than open as many tabs as you want and use `ssh princeOffCampus` to connect to the server with one call. 

## SSH Folder Mounting
Another very useful thing that I discovered recently over ssh is mounting a folder in remote server on to your local system and work with the remote folder as if it is in your computer and everything is synced automatically. To do that you need to first install this two packages.

```
brew cask install Caskroom/cask/osxfuse 
brew install sshfs
```

Now we ready to go. `sshfs` needs a symbolic folder to be created so we create that. After that I am mounting the folder(`/home/ue225/lecture1`) on remote server `stampede` to my local folder `/Users/evcu/dummy` as `customName`. On terminal the content of `lecture1` is copied to `dummy` folder, however when you open `Finder` you would see `customName` appears as remote device name. Modified the files as you wish and enjoy the magical sync happening lightning fast. Once you done you can unmount with `umount`.

```
mkdir /Users/evcu/dummy
sshfs -p 22 stampede:/home/ue225/lecture1 /Users/evcu/dummy -oauto_cache,reconnect,defer_permissions,noappledouble,negative_vncache,volname=customName,transform_symlinks,follow_symlinks
ls /Users/evcu/dummy #ls lecture1 folder. 
umount /Users/evcu/dummy
```