# nnvpy

nnvpy is the Python implementation of the nnv matlab packabe for fast verification of deep neural networks.

### Install

1) `pip install -r requirements.txt`
2) Complete extra steps for `gurobipy` setup
    1) Obtain a license for `gurobipy` and activate using `grbgetkey` (You'll have to download gurobi install files from website to access grbgetkey as that's not installed using pip)
    2) Copy the gurobi.lic file wherever you initially installed it to the following directory: [your python dir]/site-packages/gurobipy/.libs (note: if there is an existing restricted install license in the directory, simply replace it.)