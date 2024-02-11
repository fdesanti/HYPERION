
import os 
import glob
import h5py
from optparse import OptionParser

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import multiprocessing
from natsort import natsorted

from injection import Injection


if __name__ == '__main__':
    def generate_whitened_waveform(n_sim):    
        #n_sim       = n_sim#df_sources_parameters['id']==source_name
        source_name = df_id[n_sim]

        if debug:
            print('source name ', source_name)
            print('mask', n_sim)

        #extract parameters value
        m1     = df_m1[n_sim]
        m2     = df_m2[n_sim]
        dist   = df_dist[n_sim] #in pc
        p_0    = df_p_0[n_sim]  # in M unit
        e0     = df_e0[n_sim]
        t0_p   = df_t0_p[n_sim]
        ra     = df_ra[n_sim]
        dec    = df_dec[n_sim]
        pol    = df_pol[n_sim]
        incl   = df_incl[n_sim]
        time_shift = df_time_shift[n_sim]
        
        if debug:
            print(e0, p_0, m1, m2, dist, incl, pol )
            
        
        whitened_strain, _ = injector(m1, m2, dist, p_0, e0, t0_p, ra, dec, pol, incl, time_shift, return_template=return_template)
        
        return source_name, whitened_strain
    
    
    
    usg = "\033[1;31m%prog [ options] FT1FileName\033[1;m \n"
    desc = "\033[34mGenerates the Training Dataset\033[0m"
    
    parser = OptionParser(description=desc, usage=usg)
    parser.add_option("-o", "--output", default=None, help="Output Directory")
    parser.add_option("-d", "--debug", default=False, action="store_true", help="Debug mode")
    parser.add_option("-m", "--multiprocess", default=False, action="store_true", help="Use Multiprocessing")
    parser.add_option("-w", "--overwrite", default=False, action="store_true", help="Overwrites already simulated files")
    parser.add_option("-t", "--return_template", default=False, action="store_true", help="Returns the whitened templates from Injection instead of strain")

    

    (options, args) = parser.parse_args()

    #output_dir = options.output
    debug = options.debug
    multiprocess = options.multiprocess
    return_template = options.return_template
    overwrite = options.overwrite
    if debug:
        assert multiprocess == False, 'I cannot debug when multiprocessing on'
    
    parameters_sources_directory = args[0]
    parameters_sources_filepath = natsorted(glob.glob(parameters_sources_directory+'/*.hdf5'))
    print(parameters_sources_filepath)
    if len(parameters_sources_filepath)==0: #not os.path.exists(parameters_sources_filepath):
        raise FileNotFoundError("Sources parameters file does not exists: please provide a valid path")
    
    if options.output is not None:
        output_dir = options.output
    else:
        parent_dir = parameters_sources_directory#os.path.split(parameters_sources_filepath)
        parent_dir, _ = os.path.split(parent_dir)
        parent_dir, _ = os.path.split(parent_dir)
        if return_template:
            output_dir    = os.path.join(parent_dir, 'whitened_templates')
        else:
            output_dir    = os.path.join(parent_dir, 'whitened_sources')
        
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print('Output dir creadted!')
    
    print('Output files will be saved at this location: %s\n'%output_dir)
    
    if 'BHBH' in parameters_sources_directory:
        fs = 2048
    else:
        fs = 4096
  
    injector = Injection(fs = fs)
    
    for i, parameters_file in enumerate(parameters_sources_filepath):
        #if i < 2:
        #     continue
        print(f'\n----> Processing file {i+1}/{len(parameters_sources_filepath)}')
        print(parameters_file)
        #######################################################################################################
        # First, read the hdf file and store parameters in a pandas DataFrame
        df_sources_parameters = pd.read_hdf(parameters_file, mode='r', key='df_sources_parameters')
    
        #extract parameters value
        df_id         = df_sources_parameters['id'].to_numpy()
        df_m1         = df_sources_parameters['m1'].to_numpy()
        df_m2         = df_sources_parameters['m2'].to_numpy()
        df_dist       = df_sources_parameters['distance'].to_numpy() #in pc
        df_p_0        = df_sources_parameters['p_0'].to_numpy()  # in M unit
        df_e0         = df_sources_parameters['e0'].to_numpy()
        df_t0_p       = df_sources_parameters['tp'].to_numpy()
        df_ra         = df_sources_parameters['ra'].to_numpy()
        df_dec        = df_sources_parameters['dec'].to_numpy()
        df_pol        = df_sources_parameters['polarization'].to_numpy()
        df_incl       = df_sources_parameters['inclination'].to_numpy()
        df_time_shift = df_sources_parameters['time_shift'].to_numpy()
    
        if debug:
            print(df_sources_parameters)
            print('DataFrame correctly loaded')
        
        N = len(df_sources_parameters)

    
    
    # defining parallel processes for generating waveforms
    

        if not overwrite:
            #n_already_simulated = len(os.listdir(output_dir))
            hf = h5py.File(os.path.join(output_dir, 'dataset'+str(i)+'.hdf5'), 'a')
            n_already_simulated = len(hf.keys())
            print('>>>>>>> already simulated', n_already_simulated)
            if n_already_simulated == N:
                continue
        else:
            hf = h5py.File(os.path.join(output_dir, 'dataset'+str(i)+'.hdf5'), 'w')
            n_already_simulated = 0
    
    
        if multiprocess:
            n_proc = 50#2*os.cpu_count()//3
            print(f"----> using Multiprocessing with {n_proc} cores...\n")
            p = multiprocessing.Pool(processes=n_proc)
        
       
            for source_name, whitened_strain in tqdm(p.imap(generate_whitened_waveform, range(n_already_simulated ,N)), total=N-n_already_simulated):
                group = hf.create_group(source_name)
                #print(source_name)
                for det_name in whitened_strain.keys():
                    dataset = group.create_dataset(det_name, data = whitened_strain[det_name], dtype = 'float32')
                
            p.close()
            p.join()
        else:
    
            for n_sim in tqdm(range(n_already_simulated, N)):
                source_name, whitened_strain = generate_whitened_waveform(n_sim)
                group = hf.create_group(source_name)
                for det_name in whitened_strain.keys():
                    dataset = group.create_dataset(det_name, data = whitened_strain[det_name], dtype = 'float32')
    
        hf.close()
        if i+1 < len(parameters_sources_filepath):
            time.sleep(180) #let cpu and disk rest for a while

    print('\nAll waveform have been projected and whitened!')
    
    
