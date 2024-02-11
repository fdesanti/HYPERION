'''
import sys
#add to pythonpath search modules the location of the gwskysim package
gwskysim_path = ['/Users/federicodesanti/desktop/close_encounters_thesis/gwskysim', 
                  '/mnt/c/Users/fdesa/desktop/close_encounter_thesis_project/gwskysim', 
                  '/home/gwmluser/work/gwskysim',
                 ]
for path in gwskysim_path:
    sys.path.append(path)
'''
"""
Class that handles the pytorch dataset during training
"""
import os 
import sys
import glob
import h5py
import torch
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from natsort import natsorted



class DatasetManager():
    """
    Class to manage the train and validation datasets.
    
    Given the dataset population directory:
     - splits the dataset into training / validation according to train_split
     - creates iterable datasets calling the SingleWavefromDataset class

     Args:
        - dataset_directory (path): path to the desired population dataset
        - train_split (float): percentage of dataset to consider for training (default=0.9)
        - random_seed (float): seet to pass do Dataloader to shift dataset samples in a reproducible way (default = 123)
    """

    def __init__(self, 
                 dataset_directory = None, 
                 train_split = 0.9, #mutually exclusive with test
                 test = False,
                 ):
        
    
        assert train_split <= 1, 'Train split percentages must <= 1!'

        if not os.path.exists(dataset_directory):
            raise FileNotFoundError("Directory does not exists. Please give a valid path")

        self.dataset_dir   = dataset_directory
        self.train_split   = train_split
        self.test          = test


        self._split_dataset()
        
        return
    
  
    def _split_dataset(self):
        

        self.dataset_files = natsorted(glob.glob(self.dataset_dir+'/whitened_sources/*.hdf5'))

        num_files = len(self.dataset_files)
        
        #PARAMETER DF ==================================
        parameter_files = natsorted(glob.glob(self.dataset_dir+'/sources_parameters/*.hdf5'))[:num_files]

        parameter_df = pd.DataFrame()
        for parameter_file in parameter_files:
            #parameter_df   = pd.read_hdf(parameter_file, key='df_sources_parameters')
            parameter_df = pd.concat([parameter_df, pd.read_hdf(parameter_file, key='df_sources_parameters')])
        self.parameters_bounds = pd.read_hdf(parameter_files[0], key='metadata')

        #SOURCES FILES ================================
        #sim_files = sorted(glob.glob(self.dataset_dir+'/whitened_sources/*.hdf5'))

        num_sources = len(parameter_df)

        n_train = int(self.train_split * num_sources)
        
        
        if self.test:
            self.test_parameter_df = parameter_df
        else:
            self.train_parameter_df = parameter_df[:n_train]
            self.train_parameter_df = self.train_parameter_df.reset_index()
        
            self.val_parameter_df   = parameter_df[n_train:]
            self.val_parameter_df   = self.val_parameter_df.reset_index()
       
        
        return



#######################################################################################################################
#=====================================================================================================================#
#######################################################################################################################




class SimpleSingleWaveformDataset(Dataset):
    def __init__(self, 
                 parameters_df  = None, 
                 sources_fpaths = None,
                 parameters_bounds = None,
                 fs = 4096//2, 
                 duration = 1, 
                 detectors = ['L1', 'H1', 'V1'],
                 ):
        super(SimpleSingleWaveformDataset, self).__init__()

        self.sources_fpaths = sources_fpaths
        self.fs = 2048 if 'BHBH' in self.sources_fpaths[0] else 4096
        self.duration = duration
        self.detectors = np.array(detectors)
        self.num_detectors = len(detectors)
        
        self.parameters_df    = parameters_df
        self.parameters_bounds = parameters_bounds

        self.ids = self.parameters_df['id'].to_numpy()


        #converting deg --> rad
        for par_name in ['polarization', 'inclination', 'dec', 'ra']:
            self.parameters_df[par_name] = np.deg2rad(self.parameters_df[par_name])
            
            if self.parameters_bounds is not None:
                #print(par_name, self.parameters_bounds[par_name])
                self.parameters_bounds[par_name]['min'] = np.deg2rad(self.parameters_bounds[par_name]['min'])
                self.parameters_bounds[par_name]['max'] = np.deg2rad(self.parameters_bounds[par_name]['max'])
                #print(par_name, self.parameters_bounds[par_name])

        
        #sorting masses 
        self.sort_masses()
        

        #chirp mass & q
        self.compute_chirp_mass_and_q()
      
        if self.parameters_bounds is not None:
            self.parameters_bounds['q'] = {
                'max': self.parameters_bounds['m2']['max'] / self.parameters_bounds['m1']['min'],
                'min': self.parameters_bounds['m2']['min'] / self.parameters_bounds['m1']['max']}
        
        #total mass
        self.parameters_df['M_tot'] = self.parameters_df['m1'].to_numpy() +self.parameters_df['m2'].to_numpy()
        if self.parameters_bounds is not None:
            self.parameters_bounds['M_tot'] = {'max': self.parameters_bounds['m1']['max'] + self.parameters_bounds['m2']['max'],
                                          'min': self.parameters_bounds['m1']['min'] + self.parameters_bounds['m2']['min']}
        
            
        #get means and stds from df
        self.means, self.stds = self.means_and_stds
        
        #open the dataset file. NB: it will be closed when the workers are shutted down after each epoch
        self.hf = []
        for path in self.sources_fpaths:
            self.hf.append(h5py.File(path, 'r'))
        return
    
    def __len__(self):
        return len(self.parameters_df)

    def close_hf(self):
        for hf in self.hf:
            hf.close()
        return
    
    @property
    def out_parameters_names(self):
        
        if 'NSNS' in self.sources_fpaths[0]:
            return np.array(['M_tot', 'q', 'e0', 'p_0','distance', 'time_shift'])
        else:
            return np.array(['M_tot', 'q', 'e0', 'p_0','distance', 'time_shift', 'ra', 'dec'])



    @property
    def means_and_stds(self):
        """deterimines the means and standard deviations for the parameters """
        means = dict()
        stds  = dict()
        
        for par_name in self.out_parameters_names:
           
            means[par_name] = [self.parameters_df[par_name].mean()]
            stds[par_name]  = [self.parameters_df[par_name].std()]

        means = pd.DataFrame.from_dict(means)
        stds  = pd.DataFrame.from_dict(stds)
        return means, stds
    

    def read_waveform(self, source_name):
        """reads whitened strain from hdf5 file and returns it as a tensor of shape [N_detectors, Len_strain]"""
        strain = np.zeros((self.num_detectors, self.fs*self.duration))
        
        file_index = int(source_name[3:])//int(1e6)
        for i, det_name in enumerate(self.detectors):
            self.hf[file_index][source_name][det_name].read_direct(strain[i])
        return torch.from_numpy(strain)
    
   
    def sort_masses(self):
        """given m1/m2 masses array returns the array sorted so that m1>m2"""
        m1 = self.parameters_df['m1'].to_numpy()
        m2 = self.parameters_df['m2'].to_numpy()
        
        m = np.array([m1, m2]).T  #has shape [N, 2]
        m_sort = np.sort(m).T     #has shape [2, N] where [0,N] < [1, N]
        m1_sort = m_sort[1]
        m2_sort = m_sort[0]
        self.parameters_df['m1'] = m1_sort
        self.parameters_df['m2'] = m2_sort
        return 
    
    
    def compute_chirp_mass_and_q(self):
        m1 = self.parameters_df['m1'].to_numpy()
        m2 = self.parameters_df['m2'].to_numpy()
        Mc = (m1*m2)**(3/5)/(m1+m2)**(1/5)
        q = m2/m1
        self.parameters_df['M_chirp'] = Mc
        self.parameters_df['q']      = q
        return 
        
    
    
    def standardize_parameters(self, parameters):
        out_pars = np.zeros(len(self.out_parameters_names))
        
        for i, key in enumerate(self.out_parameters_names):
            out_pars[i] = (parameters[key] - self.means[key][0])/self.stds[key][0]

        return torch.from_numpy(out_pars)

    def apply_new_shift(self, strain, old_time_shift, idx):
        new_time_shift = np.random.uniform(self.parameters_bounds['time_shift']['min'], self.parameters_bounds['time_shift']['max'])
        roll_factor = int((new_time_shift - old_time_shift)*self.fs*0.5)
        return strain.roll(roll_factor), new_time_shift
    
    def __getitem__(self, idx):
        #filepath = self.sources_fpaths[idx]
        source_name = self.ids[idx]
        strain = self.read_waveform(source_name)
        parameters = self.parameters_df.iloc[idx]
        #if not idx in [4260, 2061, 1413, 7382, 9916]:
        strain, new_time_shift = self.apply_new_shift(strain, parameters['time_shift'], idx)
        parameters['time_shift'] = new_time_shift

        standardized_parameters = self.standardize_parameters(parameters)
        
        return standardized_parameters.float(), strain.float()
        
        











######################################
################ TEST ################
######################################

source_type = sys.argv[1]

if __name__ == '__main__':
    import glob
    work_dir = os.getcwd()
    parent_dir, _ = os.path.split(work_dir)
    work_dir = os.path.join(parent_dir, 'fdesanti-work1')
    
    #work_dir = os.path.join(work_dir, 'dataset')
    dataset_dir = glob.glob(work_dir+'/GWsim_'+source_type+'*')[0]

    data_manager = DatasetManager(dataset_directory=dataset_dir, train_split=0.9)

    train_files = data_manager.dataset_files#sorted(glob.glob(dataset_dir+'/whitened_sources/*.hdf5'))[0:2]
    val_files   = data_manager.dataset_files#sorted(glob.glob(dataset_dir+'/whitened_sources/*.hdf5'))[0:2]
    
   
    train_parameter_df = data_manager.train_parameter_df
    val_parameter_df   = data_manager.val_parameter_df
    print(train_parameter_df)

    print('done')
    training_data = SimpleSingleWaveformDataset(sources_fpaths=train_files, parameters_df=train_parameter_df,
                                                parameters_bounds=data_manager.parameters_bounds)
    print(training_data.parameters_bounds)

    BATCH_SIZE = 512
    data_loader_kwargs = {'batch_size': BATCH_SIZE, 'pin_memory': False,
                            'num_workers':16, 'prefetch_factor': 2,
                            'drop_last': True, 
                            'shuffle' :  True}
    
    
    '''
    train_loader = DataLoader(training_data,
                              #sampler = SubsetRandomSampler(torch.randint(high=len(train_files), size=(1800*BATCH_SIZE,))),
                              sampler = None,
                              **data_loader_kwargs)   
    '''
    #train_df = iter(train_loader)
    #val_df = iter(train_loader)
    import time
    a = 0
    N = int(6e5)
    imax = len(training_data) -N
    for epoch in range(10):
        start = np.random.randint(0, imax)
        indices = np.arange(start, start+N)
        subset = Subset(training_data, indices)
        train_loader = DataLoader(subset, **data_loader_kwargs)  
        print(start)
        for x, y in tqdm(train_loader, total=len(train_loader)):
            a+=1
            time.sleep(0.1)
            #pass
        

    print(x, y)
        
    #, strain.shape)
    #print(pars)


    







        


    





