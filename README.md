### Overview
*rcwa_tf* provides a library of functions to support running rigorous coupled-wave analysis (RCWA) simulations in TensorFlow, enabling optimization of complex photonic structures using automatic differentiation. To handle the complex, degenerate eigenproblems often encountered in RCWA simulation, we extend TensorFlow's *tf.linalg.eigh()* function to support complex eigenvalues and apply a Lorentzian broadening technique to regularize the gradient calculation. With a full implementation of RCWA, we support user-defined loss functions and batch optimizations over varying wavelength, wavevector, and polarization. We provide a few standard parameterizations for the unit cell that are all differentiable, including scatterers based on cylinders, elliptical posts, rectangular posts, as well as continously varying permittivity at each pixel within the unit cell. This makes the framework applicable to a wide range of devices, including photonic crystals and grating filters. By coupling the RCWA solver to a TensorFlow implementation of the band-limited angular spectrum method, we also support full metasurface optimizations comprising multiple unit cells that are accurate to within the local phase approximation.

At the core of the code is the base tensor shape *`(batchSize,  pixelsX, pixelsY, Nlayer, Nx, Ny)`*. *Nx* and *Ny* represent the points in the real space cartesian grid that constitute each unit cell, *Nlayer* corresponds to the different stacked layers in the structure, *pixelsX* and *pixelsY* access the different scatterer positions of a full metasurface and are both set to 1 for a periodic simulation, and over the *batchSize* dimension the input conditions vary (e.g., polarization, wavelength, and wavevector).

For additional details, please refer to our publication discussing this work:

https://doi.org/10.1038/s42005-021-00568-6

### Using the Solver
To run your own code based on *rcwa_tf*, ensure you have either placed the files located within the *src/* folder in the same directory, have added the *src/* folder to your *PYTHONPATH* environment variable, or if one prefers, they may add the additional lines prior to the import statements in their own code:

*import sys*\
*sys.path.append(<PATH_TO_SRC_FOLDER>)*

### Examples
There are a few example IPython notebooks in the *examples/* folder that show how to set up and initialize a system and optimization, define a loss function and optimizer, and to visualize the resulting data. These examples were developed to run on Google Colab.
