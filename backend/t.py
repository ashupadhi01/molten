import os
import time
import sys

print(f"process id: ", os.getpid())

def _get_memory_usage():
    
    
    with open(f"/proc/{os.getpid()}/smaps_rollup") as f:
        sum(
            int(line.split()[1])
            for line in f
            if line.startswith(("Priave_Clean", "Private_Dirty"))
        )
                
while True:
    _get_memory_usage()


"""
I am investigating some issue with the generator method. I want to understand what is happening. I will give you a complete suite of observations.

I have wrote test() method in the entry point which runs a infinite loop for token generation for handy debugging. 

The first thing I am confused about is the following:
When I run a normal python file. It runs a single process,. i.e., the root process. But when I am running my `generator.py` it is spawning the root process along with its multiple child process or threads, which I am confused about. This is happening before I am firing the .generate() method of CustomGenerator class. 

The pstree -ap <pid> shows a graph something like this when I run the generator file only:

```
~ > pstree -ap 53710

python3,53710 generator.py true
  ├─{python3},53711
  ├─{python3},53712
  ├─{python3},53713
  └─{python3},53752
```

May they call these threads. But as far as I know python is a single threaded system and I am not even spawning uvicorn that it will spawn multiple worker sub-process. I am simply running using command uv run <file_name.py> here.

After giving a sequence length and prompt when I fire the .generate() of my custom class the process tree suddenly starts to look like this:

```
~ > pstree -ap 53710
python3,53710 generator.py true
  ├─{python3},53711
  ├─{python3},53712
  ├─{python3},53713
  ├─{python3},53752
  ├─{python3},54323
  ├─{python3},54324
  ├─{python3},54325
  ├─{python3},54326
  └─{python3},54327
```

I don't understand for where these extra process or threads, I don't know came from.
First discuss this and then I will talk about the next thing I am confused about.


Let's 

"""