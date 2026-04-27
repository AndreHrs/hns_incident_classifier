## pre-requisite
Install jupytext, following is the example for using conda
```bash
conda install jupytext -c conda-forge 
```

Can be ignored since the main_notebook is already set as paired notebook
```bash
jupytext --set-formats ipynb,py:percent main_notebook.ipynb 
```

To synchronize notebook to paired .py call
```bash
jupytext --sync main_notebook.ipynb
```

To synchronize notebook from paired .py call
```bash
jupytext --sync main_notebook.py
```