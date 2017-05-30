---
layout: single
title: "SSH Port Forwarding and Jupyter Notebook"
tags: [tutorial,basics,ssh,hpc]
category: notes
excerpt: "port forwarding with ssh."
---
In the last few years Jupyter notebooks became a vital part of my coding routine. I first started using jupyter notebooks as a python interpreter and run it locally to try out some basic things. Than I learned that one can run the notebook in a server and open it locally in the browser. With this one can use the power of a remote server as if working on a jupyter notebook locally. 

In this post I will explain three things
- How to start a jupyter notebook on a server using local forwarding.
- How to configure ssh-tunelling.
- How to start jupyter notebook-job.

## Jupyter Notebooks with Basic Local Forwarding.
When you run 
> jupyter notebook

An notebook is created on the default port, which is `http://localhost:8888/` on my Mac. One can use `--port ****` flag to specify the port of the local host. If we want to run the notebook on a server through ssh, we can use `Local Forwarding` which is the `-L` flag. Lets assume we have a server named  `hostone` defined in `~/.ssh/config/`. For more information about how to do that you can check out my [ssh notes](dummylink).

```
ssh -L 8000:localhost:8888 hostone
jupyter notebook --port 8888
```

Running the two lines above would start a notebook on the server `hostone` at port 8888. And local forwarding connects the local port 8000 to the port 8888 of the server. 

## SSH Tunneling. 
Sometimes connecting to a server requires two ssh connections if you are outside of the local network. This was the case when I worked off-campus and wanted to use NYU's clusters. Having two ssh connections makes local forwarding above a little more complicated. Without any tunneling one would need to arrange two local forwarding carefully. Another annoying thing when you work off-campus is that you need to type two ssh connection everytime you wanna open a new shell. Tunneling fixes both of the problems above. This part is inspired from the [link](dummy). 

I will be using NYU HPC off-campus access example to explain tunneling. If you are off-campus one needs to an ssh call to `hpc.nyu.edu` first and than another one to `prince.hpc.nyu.edu`. Lets start with adding following lines to our `~/.ssh/config` file to define our tunnel. 

```
Host hpc2tunnel
  HostName hpc.nyu.edu
  ForwardX11 no
  LocalForward 8026 prince.hpc.nyu.edu:22
  LocalForward 8025 dumbo.hpc.nyu.edu:22
  User ue225
```

This defines the tunnel, which is nothing but an ssh connection with local forwarding connecting localhost:8026 to `prince.hpc.nyu.edu` on `hpc.nyu.edu`. . (WHHY 22?). Other local forwarding definitions can be defined on the same Host definition. After this when `ssh hpc2tunnel` called our localhost at port 8026 listens `prince.hpc.nyu.edu:22` on `hpc.nyu.edu` therefore we can ssh to our localhost at port 8026 to connect to the prince cluster with one call. Adding the definition below does this for us.
```
Host princeOffCampus
  HostName localhost
  Port 8026
  ForwardX11 yes
  User ue225
```

After these definitions all you need to do call `ssh hpc2tunnel` and leave it open. Than open as many tabs as you want and use `ssh princeOffCampus` to connect to the server at once. 

- __Running a jupyter notebook on a Slurm cluster__
 