# Functional AIES

Setup:

- Requires Python 3.6+ (uses f-strings and the `@` operator)
- Install packages in `requirements.txt`
- To compile: `gcc -fPIC -shared -o euler_solver.so src/euler_solver.c`

If C99 error: `gcc -std=c99 -fPIC -shared -o euler_solver.so euler_solver.c`


## Get samples used in the paper

Download the zip file from [here](https://lwr-inverse-mcmc.s3.eu-west-2.amazonaws.com/FES_outputs/outputs.zip), unzip it, and put it in root directory.


## scdripts

### Advection example
- `advection_standard.py`: run the standard pCN sampler on the advection example
- `advection_AIES.py`: run FES on the advection example. Modify the arguments in the `run_MCMC()` function at the bottom of the script to try different tuning parameters `M`. The corresponding omega parameters are given in lines 183-188
- `advection_figures.py`: print out the IAT values for the different values and create the ACF plot

### Langevin example
- `langevin_pCN_sampler.py`: run the standard pCN sampler for the Langevin example. This uses a previous run to tune the covariance proposal for alpha and sigma
- `langevin_hybrid_sampler.py`: run the hybrid sampler. This uses a covariance matrix fit to a previous pCN run
- `langevin_ensemble_joint_update.py`: run FES with a joint update. Change the number of walkers (`L`) at the top of the script
- `langevin_ensemble_MwG.py`: run FES with a MwG update.
- `langevin_figures_IAT_ACF.py`: print out the IAT values for the different values and create the ACF plot
- `langevin_figures_hybrid_and_posterior.py`: create plot showing the slow adaptation of the hybrid sampler along with the posterior plot for alpha, sigma, and X_t paths.
