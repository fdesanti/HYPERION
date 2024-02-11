#!/usr/bin/env python
import sys
#add to pythonpath search modules the location of the gwskysim package
gwskysim_path = ['/Users/federicodesanti/desktop/close_encounters_thesis/gwskysim', 
                  '/mnt/c/Users/fdesa/desktop/close_encounter_thesis_project/gwskysim', 
                 #'\Users\fdesa\desktop\close_encounter_thesis_project\gwskysim-master'
                  '/home/gwmluser/work/gwskysim',
                 ]
for path in gwskysim_path:
    sys.path.append(path)

"""

 Script name: generate_sources_parameters

 Description:
 Python script that generate parameters for each source of a GW population from a config JSON file.
 Always run this script in your work directory and not inside the population directory. If a population directory
 already exists and you want to increase the number of sources, just specify the name of the existing population
 directory in the configuration file (as value for key "output_dir"). On the contrary if you want a new population
 just write "null" in the configuration file or specify a new population name.

"""
import json
import numpy as np
import os


import platform
#import sys
import time
from optparse import OptionParser
import pandas as pd
import tqdm

# custom imports
from gwskysim.utilities.gwlogger import GWLogger
from gwskysim.utilities.gwsim_util import random_from_distribution

# Now, some additional information

__author__ = "Massimiliano Razzano"
__copyright__ = "Copyright 2016-2020, Massimiliano Razzano"
__credits__ = ["Line for credits"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "M. Razzano"
__email__ = "massimiliano.razzano@pi.infn.it"
__status__ = "Production"

# General variables
os_system = platform.system()
running_python_version = int(sys.version[0])

"""set work dir to fdesanti-work2"""
work_dir = os.getcwd()
parent_dir, _ = os.path.split(work_dir)
parent_dir, _ = os.path.split(parent_dir)
work_dir = os.path.join(parent_dir, 'fdesanti-work1')

script_name = os.path.split(sys.argv[0])[1].split('.')[0]
script_path = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]))

# General Functions. None here, already imported from GeneralUtilities

#######################################################
# Main
#######################################################

if __name__ == '__main__':

    usg = "\033[1;31m%prog [ options] FT1FileName\033[1;m \n"

    desc = "\033[34mPrepare the files for gw simulation\033[0m"

    parser = OptionParser(description=desc, usage=usg)
    parser.add_option("-o", "--output", default=None, help="Output Directory")
    parser.add_option("-d", "--debug", default=False, action="store_true", help="Debug mode")
    parser.add_option("-t", "--test", default=False, action="store_true", help="Test dataset")

    (options, args) = parser.parse_args()

    output_dir = options.output
    debug = options.debug
    test  = options.test

    config_filename = None

    #####################################################

    start_time = str(time.strftime("%y/%m/%d at %H:%M:%S", time.localtime()))

    # Start the logger and check if debug mode is on
    my_logger = GWLogger("generate_pop_logger")
    my_logger.set_loglevel("INFO")

    if debug:
        my_logger.set_loglevel("DEBUG")

    my_logger.info('**************************************')
    my_logger.info("    " + str(script_name))
    my_logger.info("     (Running on " + os_system + " OS)")
    my_logger.info('**************************************')

    if len(args) == 0:
        my_logger.fatal("No JSON Configuration file specified!")
        my_logger.fatal('Type ' + str(script_name) + '.py -h for help\n')
        sys.exit(1)
    else:
        config_filename = args[0]

    # List parameters and args
    my_logger.info('\nInput options:')
    for key, val in iter(parser.values.__dict__.items()):
        my_logger.info(key + ": " + str(val))

    my_logger.info("Working directory: " + work_dir)

    # First, read the JSON
    with open(config_filename) as json_file:
        pop_config_data = json.load(json_file)

    population_name = pop_config_data["population_name"]
    
    if output_dir is None:
        output_dir = pop_config_data["output_dir"]
        prefix = "GWsim_"
        if test:
            prefix = "Test_"+prefix
        output_dir = os.path.join(work_dir,
                                  prefix + population_name + str(time.strftime("_%Y%m%d", time.localtime())))
        my_logger.debug("Output directory set to %s" % output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # creating a directory for parameters files, if it already exists check inside and take the id of the last already
    # generated-signal in order to create new signals
    parameter_dir = os.path.join(output_dir, 'sources_parameters')
    if not os.path.exists(parameter_dir):
        os.mkdir(parameter_dir)
        n_first_signal = 0
    else:
        list_parameter_files = sorted(os.listdir(parameter_dir))
        last_file_name = list_parameter_files[len(list_parameter_files)-1]
        n_first_signal = int(last_file_name[-11:-5])+1

    # read the configuration parameters
    seed = pop_config_data['random_seed'] + n_first_signal
    if test:      #changing seed in case of test dataset
        seed+=123
    np.random.seed(seed)  # to allow replicable analysis
    
    if test:
        n_sources_per_file = 10000
        n_files            = 1
    else:
        n_sources_per_file = pop_config_data['sources_per_file']  # each output file will contain this number of sources
        n_files = pop_config_data['n_files']  # number of output files
    n_tot_sources = n_sources_per_file * n_files  # total number of sources
    # below we have a python dic whose keys are names of parameters, for each key the corresponding value is a dic
    # whose keys are 'distribution', 'min' and 'max' of the parameter.
    dic_par = pop_config_data['parameters']

    # loop and generate the required number of sources and fill a Pandas DataFrame, using function random_from
    # distribution defined in gwskysim.utilities.gwsim_util
    df = pd.DataFrame()
    
    for par_name, par_features in dic_par.items():
        df[par_name] = random_from_distribution(par_features['distribution'], par_features['min'],
                                                par_features['max'], n_tot_sources)
    # saving parameters distribution and ranges in a DataFrame (needed later for saving as metadata)
    dic_par = pd.DataFrame.from_dict(dic_par)
    
    # add source type column, meeded for the generation
    # and prepare id and giving each source a name
    st_list = []
    id_list = []
    for ni in range(n_first_signal, n_first_signal+ n_tot_sources):
        id_list.append("gws%.6d" % ni)
        st_list.append(pop_config_data['source_type'])

    id_series = pd.Series(np.asarray(id_list), name="id")
    st_series = pd.Series(np.asarray(st_list), name="type")
    df = pd.concat([id_series, st_series, df], axis=1)


    # split the dataframe in the given n_files number and writing the HDF5 files
    # (compression is enabled, PYtables module needed for the function to_hdf to work)
    for i_file in range(n_files):
        index_first_signal = i_file * n_sources_per_file
        index_last_signal = (i_file+1) * n_sources_per_file
        id_first_signal = df['id'][index_first_signal]
        id_last_signal = df['id'][index_last_signal-1]
        file_df = df.iloc[index_first_signal:index_last_signal]
        file_df = file_df.reset_index(drop=True)
        output_filename = "{}-{}.hdf5".format(id_first_signal, id_last_signal)
        output_filename = os.path.join(parameter_dir, output_filename)
        file_df.to_hdf(output_filename, key='df_sources_parameters', mode='w', complevel=5)
        # saving metadata
        dic_par.to_hdf(output_filename, key='metadata')

        my_logger.info("Output saved as %s" % output_filename)

        # read files for checking
        if debug:
            my_logger.debug("Reading file for debugging purposes...")
            df_test = pd.read_hdf(output_filename, mode='r', key='df_sources_parameters')
            print(df_test)
            df_meta = pd.read_hdf(output_filename, mode='r', key='metadata')
            print(df_meta)

    my_logger.info("Sources parameters files have been successfully created")
