# Functional AIES

Setup:

- Requires Python 3.6+ (uses f-strings and the `@` operator). The results in the paper use Python 3.8.5.
- Install packages in `requirements.txt`
- To compile: `gcc -fPIC -shared -o euler_solver.so src/euler_solver.c`

If you get a C99 error, run: `gcc -std=c99 -fPIC -shared -o euler_solver.so euler_solver.c`


## Get samples used in the paper

Download the zip file from [here](https://lwr-inverse-mcmc.s3.eu-west-2.amazonaws.com/FES_outputs/outputs.zip), unzip it, and put it in root directory.


## scripts

### Advection example

#### Runs

- `advection_standard.py`: run the standard pCN sampler on the advection example
- `advection_AIES.py`: run FES on the advection example. Modify the arguments in the `run_MCMC()` function at the bottom of the script to try different tuning parameters `M`. The corresponding omega parameters are given in lines 183-188

#### Figures

- `generate_advection_IAT_ACF_figures.py`: print out the IAT values for the different values and create the ACF plot
- `generate_advection_figure_conditional_sampling.py`: generate figure with conditional samples: `rho_0 | c`

### Langevin example

#### Runs

- `langevin_pCN_sampler.py`: run the standard pCN sampler for the Langevin example. This uses a previous run to tune the covariance proposal for alpha and sigma
- `langevin_hybrid_sampler.py`: run the hybrid sampler. This uses a covariance matrix fit to a previous pCN run
- `langevin_ensemble_joint_update.py`: run FES with a joint update. Change the number of walkers (`L`) at the top of the script
- `langevin_ensemble_MwG.py`: run FES with a MwG update.

#### Figures

- `generate_langevin_figures_IAT_ACF.py`: print out the IAT values for the different values and create the ACF plot
- `generate_langevin_figures_hybrid_and_posterior.py`: create plot showing the slow adaptation of the hybrid sampler along with the posterior plot for alpha, sigma, and X_t paths.
