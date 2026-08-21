# Empirical Validation of Monte Carlo Convergence to the Black-Scholes-Merton Model of Option Pricing

## Overview
This project empirically validates the convergence of the Monte Carlo pricing estimate to the closed-form analytical Black-Scholes-Merton solution for the price of a European option. It also examines how the error rate of the Monte Carlo pricing estimator converges to zero at the expected theoretical rate. Finally, the project demonstrates how increasing the number of time steps per simulated path provides no increase in pricing accuracy, while increasing the execution time of the program.

A comprehensive look into the experiment itself, as well as in-depth analysis, is presented in the accompanying [paper](mcpaper.pdf).

## Project Structure
```text
bsm_monte_carlo/
├── .gitignore.txt
├── README.md
├── mc_dataengine.py
└── mcpaper.pdf
```

* `mc_dataengine.py`: Python program used in the paper for the Monte Carlo simulation and figure generation.
* `mcpaper.pdf`: Comprehensive look at the project, mathematics, results, and analysis.
* `README.md`: General overview and installation instructions.

## Requirements
* Python 3.12-3.14.4
* NumPy 2.5.1
* Matplotlib

## Installation
1. Clone the repository:
```bash
   git clone https://github.com/evarvachtian/Mathematics.git
   cd Mathematics/projects/bsm_monte_carlo
```
2. Create a virtual environment
```bash
   python -m venv .venv
```
3. Activate the virtual environment
   
   For Windows:
   ```bash
   .venv\Scripts\activate
   ```

   For Linux/MacOs:
   ```bash
   source .venv/bin/activate
   ```

4. Install dependencies
   
   With the virtual environment installed and activated, run:

```bash
   pip install -r requirements.txt
```

Note that NumPy 2.5.1 is recommended, since this is the version used to create the original images found within the [paper](mcpaper.pdf) and should produce the best reproducibility of results. The [project](mc_dataengine.py) was developed and tested using python 3.14.4 and NumPy 2.5.1. 

## Running the Program
Run the python script from within the project directory:
```bash
python mc_dataengine.py
```

Executing the program will run the Monte Carlo simulation and create three (3) output images located in the same folder as wherever the python file is located. The images are named plot_error, plot_convergence, and plot_time. The first two images (plot_error.png and plot_time.png) should be very similar to the ones found within the [paper](mcpaper.pdf). The final image, computation time, will not look exactly like it does in the [paper](mcpaper.png) since it will be reproduced under a different  environment. Try running `time python mc_dataengine.py` a few times and look at the "real" time. For me, the program usually executes in between 16-18 seconds, but see how different this would be on your hardware.

## Author
Eric Varvachtian

