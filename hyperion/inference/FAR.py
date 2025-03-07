import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from gwpy.timeseries import TimeSeries


from .importance_sampling import *
from ..core.utilities import HYPERION_Logger

log = HYPERION_Logger()


class DetectionStat():
    """
    Class for calculating detection statistics for gravitational wave events.

    This class processes gravitational wave data around a specified central time of the event,
    applies shifts to simulate false triggers, and computes false alarm rates using
    matched filter techniques.

    The False Alarm Rate (FAR) is computed following Usman et al. (2015) (arXiv:1508.02357)
    adopting Bayes factors as detection statistics. 



    Args:
        cenral_time     (float): GPS time of the event we want to analyze.
        observation_time (float): Duration of the whole observation in seconds. (Default is 3600)
        shift_length    (float): Length of the shifts in seconds. (Default is 0.1)
        sample_duration (float): Duration of the sample in seconds. (Default is 1)
        importance_sampler (ImportanceSampling): Importance sampler to use for the calculations.
        fs              (float): Sampling frequency of the data.
        device          (str)  : Device to use for the calculations.
    """
    def __init__(self,
                central_time,
                observation_time   = 3600,  # s
                shift_length       = 0.1,   # s
                sample_duration    = 1,     # s
                importance_sampler = None,
                fs                 = 2048,
                device             = 'cpu',
                ):
        self.central_time = central_time  # GPS time of the event we want to analyze
        self.observation_time = observation_time  # duration of the whole observation in seconds
        # TODO: could also ask for event name instead of the central time
        self.shift_length = shift_length  # length of the shifts in seconds
        self.n_tot_shift = self.observation_time/self.shift_length  # number of shifts for realizing false triggers
        
        log.info(f"Going to calculate FAR on {self.observation_time}s of data.")
        log.info(f"[INFO]: Going to have {self.n_tot_shift} shifts of {self.shift_length}s each.")

        self.sample_duration = sample_duration
        self.distrib_has_been_done = False  # boolean to check if the distribution has been calculated
        
        self.IS = importance_sampler
        self.device = device
        self.fs = fs
        self.central_idx = 0.5 * self.fs * self.observation_time

        # set global constants
        self.flight_time = {'L1_H1':0.008, 'L1_V1':0.014,
                            'H1_L1':0.008, 'H1_V1':0.014,
                            'V1_L1':0.014, 'V1_H1':0.014}
        
        # TODO: could generalize the time range opened to different ifos, for now let's take Virgo one since it's the longest
        self.time_range = [self.central_time - self.flight_time['V1_H1'] - self.n_tot_shift*self.shift_length - self.sample_duration/2
                           , self.central_time + self.flight_time['V1_H1'] + self.n_tot_shift*self.shift_length + self.sample_duration/2]
        #print(self.time_range)
        #self.total_observation_time = self.time_range[1] - self.time_range[0]
        self.total_background_time = self.observation_time**2 / self.shift_length # def of Usman et al 2016
        
        self.fails_counter = 0

        self.ifos = ['L1', 'H1', 'V1']
        self.data = {}  # get data
        self.data_white = {}  # get whitened data

        # data identifier
        # get date and time
        now = datetime.datetime.now()
        self.data_id = 'gps' + str(int(self.central_time)) + '_' + 'obstime' + str(int(self.observation_time))

    @property
    def FAR(self):
        """Getter for the false alarm rate."""
        # gives the FAR if it has been calculated, otherwise tells you to run the calculation
        if not hasattr(self, '_FAR'):
            raise ValueError('The false alarm rate has not been calculated yet. Run the calculate_false_alarm_rate() method.')
        else:
            return self._FAR
        
    @FAR.setter
    def FAR(self, value):
        """Setter for the false alarm rate."""
        self._FAR = value

    def download_data(self):
        """
        Downloads the data from the time range and saves it in a directory called 'data' in csv format.
        """
        # create a directory to save the data
        os.mkdir('data_' + self.data_id)
        os.chdir('data_' + self.data_id)
        
        start = self.central_time - self.observation_time/2
        end   = self.central_time + self.observation_time/2
        for ifo in self.ifos:
            print(f"Downloading data of {ifo} ifo...")
            
            #self.data[ifo] = TimeSeries.fetch_open_data(ifo, self.time_range[0], self.time_range[1], cache=True).resample(2048)
            self.data[ifo] = np.nan_to_num(TimeSeries.fetch_open_data(ifo, start, end, cache=True)).resample(2048)
            # save data
            self.data[ifo].write(ifo + '_' + self.data_id + '.csv', format='csv')

            # save whitened data
            self.data_white[ifo] = self.data[ifo].whiten(4, 2)
            self.data_white[ifo].write(ifo + '_' + self.data_id + '_white.csv', format='csv')

    def open_data(self):
        """
        Opens the data from the time range and returns the data.
        """
        os.chdir('data_' + self.data_id)
        for ifo in self.ifos:
            print(f"[INFO]: reading data from {ifo}")
            if ifo == 'V1':
                self.data[ifo] = TimeSeries.read(ifo + '_' + self.data_id + '.csv', format='csv')
                
                # self.data[ifo] = np.nan_to_num(self.data[ifo])
                # #asd = self.data[ifo].crop(self.central_time-32, self.central_time).asd(4,2)
            
                # whiten = self.data[ifo].whiten(4, 2).write(ifo + '_' + self.data_id + '_white.csv', format='csv')
                
                # from hyperion.fft import rfft, irfft
                # #print(len(asd))
                
                # import matplotlib.pyplot as plt
                # #plt.loglog(asd)
                # #plt.show()
                # plt.plot(whiten)
                # plt.show()
                # print(whiten)
            
            self.data_white[ifo] = torch.from_numpy(TimeSeries.read(ifo + '_' + self.data_id + '_white.csv', format='csv')).float().to(self.device)
            print('len data', len(self.data_white[ifo]))

    def calculate_false_triggers_distribution(self, event_psd, shift_mode):
        # look for the data in a directory called 'data' in csv format, if there's no data, download it
        if os.path.exists('data_' + self.data_id):
            print('Data already present ...\n')
            self.open_data()
        else:
            print('Data not present -> downloading them...')
            self.download_data()
        
        """Calculates the false triggers."""
        # shifts seconds
        shifts_pos = np.arange(self.shift_length, (self.n_tot_shift*self.shift_length - self.sample_duration)/2, self.shift_length)
        shifts_neg = - shifts_pos
        #self.stat_values = np.array([])
        #print(shifts_pos)
        # save stat at each shift in the txt file
        fname = f"stat_values_{self.n_tot_shift}_gps_time{self.central_time}_shift-mode_{shift_mode}.txt"
        with open(fname, "w") as f:
            # I have to create fake triggers by shifting two of the ifos and then calculate detection statistiss
            #f.write(f"# logB, log10B, sample efficiency, {self.IS.parameter_names} \n")
            out_str = f"logB,log10B,sample_efficiency"
            for name in self.IS.parameter_names:
                out_str += f",{name}"
            out_str += f",ks_stat"
            out_str+='\n'
            f.write(out_str)
            f.flush()
            
            for ifo in self.ifos:
                # fix one ifo, shift the other two
                # make an array with the other two ifos keys
                print(f"[INFO]: Fixing {ifo} ifo and shifting the others")
                
                ifos_shifted = [x for x in self.ifos if x != ifo]
                
                if shift_mode == 'inverted':
                    ifos_shifted = ifos_shifted[::-1] #invert shifts between detectors

                # fix data of the fixed ifo
                data_temp = {}
                data_temp_white = {}

                # check if len(data_temp) is zero, unless raise an erorr
                if len(data_temp) != 0 or len(data_temp_white) != 0:
                    raise ValueError('The data_temp dictionary is not empty. Check the code.')

                #data_temp[ifo]       = torch.from_numpy(self.data[ifo].crop(self.central_time - self.sample_duration/2, self.central_time + self.sample_duration/2).copy()).float().to(self.device)
                #data_temp_white[ifo] = torch.from_numpy(self.data_white[ifo].crop(self.central_time - self.sample_duration/2, self.central_time + self.sample_duration/2).copy()).float().to(self.device)

                
                start = int(self.central_idx - self.sample_duration*self.fs/2)
                end   = int(self.central_idx + self.sample_duration*self.fs/2)
                #print(start, end, end-start)
                data_temp_white[ifo] = self.data_white[ifo][start : end]
                
                #print('len', len(data_temp_white[ifo]), len(self.data_white[ifo]))
                
                # shift the data
                i = 0 # counter for the shifts
                
                for shift_p, shift_n in tqdm(zip(shifts_pos, shifts_neg), total = len(shifts_pos)):
                    # shift the data
                    
                    #data_temp.update(self.shift_data(self.data, ifos_to_shift=ifos_shifted, shift_p=shift_p, shift_n=shift_n, shift_index=i))
                    data_temp_white.update(self.shift_data(self.data_white, ifos_to_shift=ifos_shifted, shift_p=shift_p, shift_n=shift_n, shift_index=i))

                    #print(f'data temp: {data_temp}')
                    #print(f'[INFO] {now} - Calculating detection statistics for shift {i}/{self.n_tot_shift}...')
                    i += 1
    
                    # calculate detection statistics
                    logB, log10B, sample_efficiency, medians, ks_stat = self.calculate_detection_statistics(strain=data_temp_white,
                                                                whitened_strain=data_temp_white,
                                                                event_psd=event_psd,
                                                                gps_time=self.central_time)
                    
                    
                    out_str = f"{logB},{log10B},{sample_efficiency}"
                    for val in medians:
                        out_str += f",{val}"
                    out_str += f",{ks_stat}"
                    out_str+='\n'
                    f.write(out_str)
                    f.flush()
            # Flush the buffer
            #f.flush()
    
        # set the boolean to True
        #self.distrib_has_been_done = True
        log.info(f"[INFO] File of false triggers statistics saved at {fname}")

    def shift_data(self, data, ifos_to_shift, shift_p, shift_n, shift_index):
        #print('I am here in shift data')
        """Shifts the data by the specified amount."""
        # data to shift
        data_shifted = {}
        #print(f'Data in shift_data: {data}')

        #print(shift_p, shift_n)
        # for even shifts, shift ifo 0 forward and ifo 1 backward
        # for odd shifts, shift ifo 0 backward and ifo 1 forward
        '''
        if shift_index%2 == 0:
            data_shifted[ifos_to_shift[0]] = torch.from_numpy(data[ifos_to_shift[0]].crop(
                self.central_time - self.sample_duration/2 + shift_p ,
                self.central_time + self.sample_duration/2 + shift_p ).copy()).float().to(self.device)
            data_shifted[ifos_to_shift[1]] = torch.from_numpy(data[ifos_to_shift[1]].crop(
                self.central_time - self.sample_duration/2 + shift_n  ,
                self.central_time + self.sample_duration/2 + shift_n  ).copy()).float().to(self.device)
        else:
            data_shifted[ifos_to_shift[0]] = torch.from_numpy(data[ifos_to_shift[0]].crop(
                self.central_time - self.sample_duration/2 + shift_n ,
                self.central_time + self.sample_duration/2 + shift_n ).copy()).float().to(self.device)
            data_shifted[ifos_to_shift[1]] = torch.from_numpy(data[ifos_to_shift[1]].crop(
                self.central_time - self.sample_duration/2 + shift_p ,
                self.central_time + self.sample_duration/2 + shift_p ).copy()).float().to(self.device)
        '''
        
        if shift_index%2 == 0:
            start_0 = int(self.central_idx + shift_p * self.fs - 0.5*self.sample_duration * self.fs)
            end_0   = int(self.central_idx + shift_p * self.fs + 0.5*self.sample_duration * self.fs)
            start_1 = int(self.central_idx + shift_n * self.fs - 0.5*self.sample_duration * self.fs)
            end_1   = int(self.central_idx + shift_n * self.fs + 0.5*self.sample_duration * self.fs)
        else:
            start_0 = int(self.central_idx + shift_n * self.fs - 0.5*self.sample_duration * self.fs)
            end_0   = int(self.central_idx + shift_n * self.fs + 0.5*self.sample_duration * self.fs)
            start_1 = int(self.central_idx + shift_p * self.fs - 0.5*self.sample_duration * self.fs)
            end_1   = int(self.central_idx + shift_p * self.fs + 0.5*self.sample_duration * self.fs)
        
        data_shifted[ifos_to_shift[0]] = data[ifos_to_shift[0]][start_0 : end_0]
        data_shifted[ifos_to_shift[1]] = data[ifos_to_shift[1]][start_1 : end_1]
        #print(data_shifted)
        return data_shifted

    def calculate_detection_statistics(self, strain, whitened_strain, event_psd, gps_time):
        """Calculates the detection statistics."""
        # get the bayes factor
        #print('whitened strain', whitened_strain['L1'].shape, whitened_strain['H1'].shape, whitened_strain['V1'].shape)
        #try:
        
        #import matplotlib.pyplot as plt
        #plt.plot(whitened_strain['V1'].cpu().numpy())
        #plt.show()
        try:
            logB, log10B, sample_efficiency, medians, ks_stat = self.IS.compute_Bayes_factor(whitened_strain, whitened_strain, event_psd, gps_time)
        
        
        except Exception as e:
            print('WARNING: something went wrong with IS')
            print(e)
            self.fails_counter += 1
            return np.nan, np.nan, np.nan, [np.nan for _ in self.IS.parameter_names], np.nan
        
        
        return logB, log10B, sample_efficiency, medians, ks_stat

    def statistic_histogram(self, bins, plot=False):
        """Returns the histogram of the statistic."""
        # check if self.stat exists
        if self.stat is not None:
            ValueError('The statistic has not been calculated yet. Run the calculate_false_triggers() method.')
        else:
            if plot:
                # plot the histogram
                plt.hist(self.stat, bins=bins, density=True)
                plt.show()
            return np.histogram(self.stat, bins=bins, density=True)  # density=True gives the probability density function (norm area)

    def calculate_probability_area(self, event_stat_value, flow, event_psd, injector, bins=100):
        """Calculates the false alarm rate."""
        # set the bins
        self.bins = bins

        # check if the distribution has been calculated
        if not self.distrib_has_been_done:
            self.calculate_false_triggers_distribution(flow, event_psd, injector)

        # get the histogram
        bin_counts, bin_edges = self.statistic_histogram(self.stat_values, bins=self.bins)

        # Initialize area
        area = 0

        # Iterate over bins
        for i in range(len(bin_counts)):
            bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
            bin_width = bin_end - bin_start

            # Check if the bin is completely to the right of the threshold
            if bin_start >= event_stat_value:
                area += bin_counts[i] * bin_width
            elif bin_start < event_stat_value < bin_end:
                # Handle partial bin containing the threshold
                partial_width = bin_end - event_stat_value
                proportion = partial_width / bin_width
                area += bin_counts[i] * proportion * bin_width

        # Gives the area of the histogram to the right of the threshold
        return area
    
    def count_triggers_over_threshold(self, stat, event_stat_value, mask):
        """Returns the cumulative histogram over the threshold."""
        # background counts that are above the threshold
        #bg_count = sum(1 for i in range(len(stat)) if stat[i] > event_stat_value)
        stat_mask = (stat>event_stat_value) * mask
        bg_count = len(stat[stat_mask])
        
        print(f"background counts {bg_count} vs num shifts {len(stat)}")
        return bg_count
    
    def false_alarm_probability(self, bg_count, T_obs, Tbg):
        return 1 - np.exp(- T_obs*(1+bg_count)/Tbg)
        
    
    def calculate_false_alarm_rate(self, event_stat_value:tuple = None, obs_time = 3600):
        """Calculates the false alarm rate."""
        # check if the distribution has been calculated
        #if not self.distrib_has_been_done:
            #self.calculate_false_triggers_distribution(flow, event_psd, injector)
                  
        # calculate the probability
        Tbg = (obs_time)**2 / self.shift_length

        #read data
        logB   = np.array([])
        log10B = np.array([])
        sample_efficiency = np.array([])
        
        stat_df = []
        for shift_mode in ['forward', 'inverted']:
        
            fname = f"data_{self.data_id}/stat_values_{self.n_tot_shift}_gps_time{self.central_time}_shift-mode_{shift_mode}.txt"
            
            
            #x = np.loadtxt(fname, delimiter=',', unpack=False, comments='#')
            #print(x, x.shape)
            '''
            logB   = np.append(logB, logB_tmp[logB_tmp>=0])
            log10B = np.append(log10B, log10B_tmp[logB_tmp>=0])
            sample_efficiency = np.append(sample_efficiency, eff_tmp[logB_tmp>=0]*100)
            '''
            #df = pd.read_csv(fname)
            stat_df.append(pd.read_csv(fname))
        stat_df = pd.concat(stat_df)
        print(stat_df, stat_df.keys())
        
        logB = stat_df['logB'].to_numpy()
        log10B = stat_df['log10B'].to_numpy()
        sample_efficiency = stat_df['sample_efficiency'].to_numpy()

        ks_stat = stat_df['ks_stat'].to_numpy()
        
        M = stat_df['M_tot'].to_numpy()
        ra = stat_df['ra'].to_numpy()
        dl = stat_df['distance'].to_numpy()
        e0 = stat_df['e0'].to_numpy()
        
        #mask = (M <=180) * (M>=120) * (ra <=3.5) * (ra>=2.8) * (dl<=2300) * (e0<=0.8) * (e0>=0.75) 
        
        
        
        stat_name, stat_value = event_stat_value
        if stat_name == 'logB':
            stat = logB
        elif stat_name == 'log10B':
            stat = log10B
        elif stat_name in ['sample_efficiency', 'efficiency', 'eff']:
            stat = sample_efficiency


        # get background counts
        mask = ks_stat>=92
        bg_count = self.count_triggers_over_threshold(stat, stat_value, mask)


        '''
        plt.hist(stat, 100)
        plt.xlabel(stat_name)
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim(5, 20)
        plt.savefig(f"{stat_name}_stat.png", dpi=200)
        '''
        binsize = 0.1
        print(max(stat))
        dec = np.ones(len(stat))#stat 
        bins = np.arange(0, max(stat) + binsize, binsize)   
        count, _ = np.histogram(stat, bins=bins, weights= obs_time / Tbg*dec)
        count[np.where(count == 0)] = obs_time / (1000.*Tbg)
        plt.step(bins[:-1], count, where='post', linewidth=2, color='black', label='Search Background')
        #plt.yscale('log')
        #plt.xlim(6, max(stat))
        plt.show()
        
        
        FAR_hz = (1+bg_count) / Tbg
        
        # convert to FAR per year
        self.FAR = FAR_hz * ( 60 * 60 * 24 * 365)
        
        self.FAP = self.false_alarm_probability(bg_count, obs_time, Tbg)
        
        return self.FAR, self.FAP, logB, log10B, sample_efficiency