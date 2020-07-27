from copy import deepcopy
from scipy.interpolate import interp1d
import numpy as np
import os
import boto3, botocore

def cut_prior_means(prior_mean, data_array_str):
    """
    Returns prior mean cut based on dataset

    Parameters
    ----------
    prior_mean: ndarray
        array of prior mean of length 150
    data_array_str: str
        Name of data_array (CSV file) for flow data

    Returns
    -------
    new_prior_mean: ndarray
        Cut prior mean
    """
    data_IC_dict = {'data_array_70108_flow_shorter2.csv': (20, 80), # DS1,60t
            'test_data/Simulated_LWR_Nov2018/data_array_DelCast_flow.csv': (20, 80), # DelCastillo simulated data, DS1,60t
            'test_data/Simulated_LWR_Nov2018/data_array_DelCast_flow_RT_40.csv': (20, 80), # DelCastillo simulated data for high resolution BCs, DS1,60t
            'test_data/Simulated_LWR_Nov2018/data_array_Exp_flow.csv': (20, 80), # Exp simulated data, DS1,60t
            'test_data/Simulated_LWR_Nov2018/data_array_Exp_flow_RT40.csv': (20, 80), # Exp simulated data; high resolution BCs, DS1,60t
            'data_array_70108_flow_52t.csv': (28, 80), # DS1, 52t.
            'data_array_70108_flow_49t.csv': (31, 80), # DS1, 49t.
            'data_array_70108_flow_48t.csv': (32, 80), # DS1, 48t.
            'data_array_70108_flow_47t.csv': (33, 80), # DS1, 47t.
            'data_array_70108_flow_46t.csv': (34, 80), # DS1, 46t.
            'data_array_flow_70108_longer.csv': (0, 150), # DS1, 150t
            'data_array_70108_flow_shorter.csv': (0, 90),
            'test_data/artificial_data_array_flow_prior_BC_poisson_short.csv': (0, 20),

            'Sim_data_array_Exp_flow_40t.csv': (31, 71), # Exp, Simulated cut to 40t
            'Sim_data_array_DelCast_flow_40t.csv': (31, 71), # Exp, Simulated cut to 40t
            }
    new_prior_mean = deepcopy(prior_mean)
    if data_array_str in data_IC_dict.keys():
        lower_b, upper_b = data_IC_dict[data_array_str]
        new_prior_mean = new_prior_mean[lower_b:upper_b]
    else:
        raise ValueError("need to cut prior mean for this dataset")
    return new_prior_mean



def create_interpolate_prior_mean_fun(final_time, prior_mean_raw):
    """
    Create function to interpolate prior mean. Use cubic splines
    """
    f_outlet = interp1d(np.arange(0, final_time+1), prior_mean_raw, kind='cubic')
    return f_outlet

def load_CSV_data(path):
    """
    Load CSV from the data folder (`traffic_data`)

    Parameters
    ----------
    path: str
        Path relative to `traffic_data` folder

    Returns
    -------
    data: ndarray
        Array of data
    """
    return np.genfromtxt(os.path.join('data/traffic_data', path))


def uniform_log_prob(theta, lower, upper):
    if lower < theta < upper:
        return np.log(1/(upper - lower))
    else:
        return -1000000

def gen_run_setting_str(config_dict, running_time=None):
    Run_settings = f"""Comments: {config_dict["Comments"]}
    Directory name = {config_dict["dir_name"]}
    N_MCMC = {config_dict["N_MCMC"]}
    Lwalkers = {config_dict["Lwalkers"]}
    M_trunc = {config_dict["M_trunc"]}
    a_prop = {config_dict["a_prop"]}
    omega_outlet, omega_inlet = {config_dict["omega_outlet_list"][0]}, {config_dict["omega_inlet_list"][0]}
    thin_samples = {config_dict["thin_samples"]}
    move_probs = {config_dict["move_probs"]}
    betas = {config_dict["betas"]}
    save_to_S3 = {config_dict["save_to_S3"]}

    For untempered chains:
    Acceptance rate for AIES: {config_dict["acceptance_rateAIES"]:.1f}%
    Acceptance rate for outlet pCN: {config_dict["acceptance_ratepCN_Outlet"]:.1f}%
    Acceptance rate for inlet pCN: {config_dict["acceptance_ratepCN_Inlet"]:.1f}%
    Acceptance rate for swaps: {[round(e, 1) for e in config_dict["acceptance_rateSwaps"]]}%
    """
    if running_time is not None:
        Run_settings = Run_settings + f"\nRunning time: {running_time:.2f}s (ie: {(running_time)/(60*60):.2f} hours)\n"
    return Run_settings



def upload_chain(s3_path, local_path, bucket_name='lwr-inverse-us-east'):
    """
    Upload file to S3

    Parameters
    ----------
    s3_path: str
        Path to file in S3
    local_path: str
        Path to file on local machine
    bucket_name: str
        'lwr-inverse-us-east' is default
    """
    s3 = boto3.resource("s3")
    lwr_AIES = s3.Bucket(bucket_name)
    file_content = open(local_path, 'rb')
    lwr_AIES.put_object(Key=s3_path, Body=file_content)


def download_chain(s3_path, local_path, bucket_name='lwr-inverse-us-east'):
    """
    Download file from S3. Uses s3 bucket 'lwr-inverse-mcmc'

    Parameters
    ----------
    s3_path: str
        Path to file in S3
    local_path: str
        Path to file on local machine
    bucket_name: str
        'lwr-inverse-us-east' is latter is default)
    """
    s3 = boto3.resource("s3")
    lwr_AIES = s3.Bucket(bucket_name)
    try:
        lwr_AIES.download_file(Key=s3_path, Filename=local_path)
        print("Download successful")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
        else:
            raise



def save_chain_to_file(config_dict, samplesOutlet, samplesInlet, samplesFD, logPostList, w_num):
    """
    Save an untempered chain to text file.
    If config_dict['save_to_S3'] is True, upload the files to S3

    Parameters
    ----------
    config_dict: dict
        Dictionary of run parameters
    samplesOutlet, samplesInlet, samplesFD: ndarray
        Arrays of samples for the untempered chain (ie: for a single walker)
    logPostList: ndarray
        Array of log-posteriors for all walkers
    w_num: int
        Walker index number
    """
    outlet_path = f"outputs/{config_dict['dir_name']}/AIES_LWR-outlet-N_{config_dict['N_MCMC']}-L_{config_dict['Lwalkers']}-a_prop_{config_dict['a_prop']}-M_trunc_{config_dict['M_trunc']}-walker{w_num}.txt"
    inlet_path = f"outputs/{config_dict['dir_name']}/AIES_LWR-inlet-N_{config_dict['N_MCMC']}-L_{config_dict['Lwalkers']}-a_prop_{config_dict['a_prop']}-M_trunc_{config_dict['M_trunc']}-walker{w_num}.txt"
    FD_path = f"outputs/{config_dict['dir_name']}/AIES_LWR-FD-N_{config_dict['N_MCMC']}-L_{config_dict['Lwalkers']}-a_prop_{config_dict['a_prop']}-M_trunc_{config_dict['M_trunc']}-walker{w_num}.txt"
    logPostList_path = f"outputs/{config_dict['dir_name']}/AIES_LWR-FD-N_{config_dict['N_MCMC']}-L_{config_dict['Lwalkers']}-a_prop_{config_dict['a_prop']}-M_trunc_{config_dict['M_trunc']}-LogPostList.txt"
    np.savetxt(outlet_path, samplesOutlet)
    np.savetxt(inlet_path, samplesInlet)
    np.savetxt(FD_path, samplesFD)
    np.savetxt(logPostList_path, logPostList)

    Run_settings = gen_run_setting_str(config_dict)
    run_settings_path = f"outputs/{config_dict['dir_name']}/run_settings.txt"
    with open(run_settings_path, 'w') as f:
        f.write(Run_settings)
    if config_dict["save_to_S3"] == True:
        upload_chain(s3_path=os.path.join("AIES_LWR", outlet_path), local_path=outlet_path)
        upload_chain(s3_path=os.path.join("AIES_LWR", inlet_path), local_path=inlet_path)
        upload_chain(s3_path=os.path.join("AIES_LWR", FD_path), local_path=FD_path)
        upload_chain(s3_path=os.path.join("AIES_LWR", logPostList_path), local_path=logPostList_path)
        upload_chain(s3_path=os.path.join("AIES_LWR", run_settings_path), local_path=run_settings_path)


def save_current_samples(config_dict, iternum, currentOutlet, currentInlet, currentFD):
    num_temps, _, _ = currentOutlet.shape

    dir_path = f"outputs/{config_dict['dir_name']}/currentSamples"
    for t in range(num_temps):
        np.savetxt(os.path.join(dir_path, f"Outlet_temp{t}.txt"), currentOutlet[t,:,:])
        np.savetxt(os.path.join(dir_path, f"Inlet_temp{t}.txt"), currentInlet[t,:,:])
        np.savetxt(os.path.join(dir_path, f"FD_temp{t}.txt"), currentFD[t,:,:])

    with open(os.path.join(dir_path, "iter_num.txt"), 'w') as f:
        f.write(f"Iteration number {iternum}")

    if config_dict["save_to_S3"] == True:
        for t in range(num_temps):
            upload_chain(s3_path=os.path.join("AIES_LWR", os.path.join(dir_path, f"Outlet_temp{t}.txt")), local_path=os.path.join(dir_path, f"Outlet_temp{t}.txt"))
            upload_chain(s3_path=os.path.join("AIES_LWR", os.path.join(dir_path, f"Inlet_temp{t}.txt")), local_path=os.path.join(dir_path, f"Inlet_temp{t}.txt"))
            upload_chain(s3_path=os.path.join("AIES_LWR", os.path.join(dir_path, f"FD_temp{t}.txt")), local_path=os.path.join(dir_path, f"FD_temp{t}.txt"))
        upload_chain(s3_path=os.path.join("AIES_LWR", os.path.join(dir_path, "iter_num.txt")), local_path=os.path.join(dir_path, "iter_num.txt"))
