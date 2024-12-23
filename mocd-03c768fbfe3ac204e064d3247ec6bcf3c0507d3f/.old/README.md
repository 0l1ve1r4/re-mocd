
visulize time wasted in python:

- Create file after running

```bash
python3 -m cProfile -o output.pyprof python/src/main.py
```

- Graphical Interface 

```bash
snakeviz output_before_threads.pyprof
snakeviz output_pool_thread.pyprof
snakeviz output_multiprocessing.pyprof
``