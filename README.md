# Axion-Emulator

## ```Generate.py```
Generate.py implements an automatic routine for generating axionCAMB initial conditions and using these in an ${\it N}$-body simulation utilizing the COLA method. In addition to allowing you to run a pair of simulations using a given set of cosmological parameters (which can be done using ```run_sim```), Axion mass, Axion fraction and simulation setup, you can also choose to generate a Latin hypercube sampling (LHS) of the parameters of your choosing, which can be done using ```make_LHS```. You may then choose to probe these samples using the pipeline by running ```run_pipeline```, which picks up the LHS, inputs the parameters from it and runs the pipeline. If you wish to make your own emulator, or want to create a neat .csv file using the results fromn probing the LHS, you can use ```make_test_train```.

Below I give an example of how you can use ```Generate```to run the pipeline. 

```python
from Generate import Generate
gen = Generate(label="my_sim",
        outputfolder="output_main",
        axioncamb_folder="axionCAMB",
        cola_folder="FML/FML/COLASolver",
        h = 0.72,
        Omega_b = 0.05,
        Omega_ncdm = 0.005,
        f_axion = 0.5,
        m_axion = 1e-27,
        boxsize=250.0,
        npart=640,
        ntimesteps=30,
        zini=30.0)
gen.run_sim()
```

## test_axion_dependency.py
An example of how the emulator can be used is presented here.
