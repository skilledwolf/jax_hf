# jax-based HartreeFock solver for electron gas problems

Note: there is no documentation at this time, only one example file how this package could be used together with the (at time of writing private) package `contimod`.

## 1) Installation on Linux and MacOS

### Option 1 (preferred method for developers)

If you want to modify the package, then install it in `editable` mode. Just clone the project, navigate a terminal to the base directory and run 
```bash
$ pip install -e .
```
This will allow you to do `import jax_hf as thf` in your python code. You can uninstall the package with `pip uninstall jax_hf`.

If you would like to contribute, please use the standard git workflows.

### Option 2 (preferred method for users)

If you are sure that you will not need to modify the package, then open the terminal and run:
```bash
$ pip install git+https://gitlab.com/skilledwolf/jax_hf.git
```
or if you have an SSH key set up:
```bash
$ pip install git+git@gitlab.com:skilledwolf/jax_hf.git
```

Both installation options will allow you to do `import jax_hf as thf` in your python code. You can uninstall the package with `pip uninstall jax_hf` in both cases.


Acknowledgement: This Hartree-Fock solver was written with the help of OpenAI's ChatGPT.


**Author: Dr. Tobias Wolf**