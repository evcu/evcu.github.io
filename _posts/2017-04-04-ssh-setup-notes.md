---
layout: single
title: "SSH Setup Notes"
tags: [tutorial,basics,ssh,hpc]
category: notes
excerpt: "useful ssh setup commands for good"
---
If you are working remotely and connecting to some other machine via ssh all the time, sometimes it becomes too frustrating to enter the password every single time; if you don't know how to setup key-pairs. I didn't know this until very recently. So in this post I will share very simple ssh setup basics. Lets say you want to do following connection:

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
