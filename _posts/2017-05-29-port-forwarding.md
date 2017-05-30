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
- How to start jupyter notebook-job.

For other things like exchanging key, ssh aliasing&tunneling check [this](https://evcu.github.io/notes/ssh-setup-notes/). 
## Jupyter Notebooks with Basic Local Forwarding.
When you run 
> jupyter notebook

a notebook is created on the default port, which is `http://localhost:8888/` on my Mac. One can use `--port ****` flag to specify the port of the local host. If we want to run the notebook on a server through ssh, we can use `Local Forwarding` which is the `-L` flag. Lets assume we have a server named  `hostOne` defined in `~/.ssh/config/`. For more information about how to do that you can check out my [ssh notes](https://evcu.github.io/notes/ssh-setup-notes/)).

```
ssh -L 8000:localhost:8888 hostOne
jupyter notebook --port 8888
```

Running the two lines above would start a notebook on the server `hostOne` at port 8888. And local forwarding connects the localhost at port 8000 to the port 8888 of the server. 

- __Running a jupyter notebook on a Slurm cluster__
Here we need understand ssh forwarding&tunneling. Even though it seems pretty complicated at the beginning, [this](https://unix.stackexchange.com/a/118650) answer explains it pretty well. Great vis!. When a job is started, it runs on a different address and we need to create a tunnel from the job-server to the  cluster we are connecting(the one we submit job in). In our example the cluster is NYU's `prince` and it uses _Slurm_ scheduler. What we need to do is first submit a job, which does the forwarding, starts the jupyter notebook and informs us. The job would be like following (inspired from [here](https://wikis.nyu.edu/display/NYUHPC/Running+Jupyter+on+Prince)). Lets name it `jupyterGPU.job`.
```
#!/bin/bash
#SBATCH --job-name=jupyterTest
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=2:00:00

#Load necessary modules
module purge
module load python/intel/2.7.12

#Go to the folder you wanna run jupyter in
cd $HOME

#Pick a random or predefined port
#port=$(shuf -i 6000-9999 -n 1)
port=8765
#Forward the picked port to the prince on the same port. Here log-x is set to be the prince login node.
/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

#Start the notebook
jupyter notebook --no-browser --port $port
```

Than we submit the job

> sbatch run-jupyter.sbatch
Submitted batch job 953167

When it is completed we have the log named `slurm-953167.out` with output similar to this.
```
[I 17:41:05.035 NotebookApp] Writing notebook server cookie secret to /state/partition1/job-953167/jupyter/notebook_cookie_secret
[I 17:41:06.051 NotebookApp] Serving notebooks from local directory: /home/ue225
[I 17:41:06.052 NotebookApp] 0 active kernels 
[I 17:41:06.052 NotebookApp] The Jupyter Notebook is running at: http://localhost:8765/?token=33703785bdb10cadf4ab0645002ab373e8e966b114f05c11
[I 17:41:06.052 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```
All we need to at this point is to run 
> ssh -L 8765:localhost:8765 ue225@prince

and copy `http://localhost:8765/?token=33703785bdb10cadf4ab0645002ab373e8e966b114f05c11` to our browser. 
