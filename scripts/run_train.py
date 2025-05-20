import os
import yaml
import torch
import seaborn as sns
sns.set_theme()
sns.set_context("talk")

from optparse import OptionParser

from hyperion.training import (DatasetGenerator, 
                               get_LR_scheduler,
                               get_optimizer, 
                               Trainer)

from hyperion.config import CONF_DIR
from hyperion.core.flow import build_flow
from hyperion.core import HYPERION_Logger
from hyperion.simulation import (ASD_Sampler, 
                                  GWDetectorNetwork, 
                                  WaveformGenerator)
log = HYPERION_Logger()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--device", default=None, help="Device to run the training on. (Default is 'cuda')")
    parser.add_option("-m", "--model_name",  default='BHBH', help="Name of the model to train (preload). (Default: BHBH)")
    parser.add_option("-c", "--config", default=f'{CONF_DIR}/default_hyperion_config.yml', help="Path to the configuration file. (Default: {CONF_DIR}/default_hyperion_config.yml)")
    parser.add_option("-p", "--preload_trained", default=False, action="store_true", help="Load a pretrained model in training_results/<MODEL_NAME> directory.")
    (options, args) = parser.parse_args()
    
    PRELOAD    = options.preload_trained
    DEVICE     = options.device
    MODEL_NAME = options.model_name
    
    #check if training results dir exists
    if not os.path.exists('training_results'):
        os.mkdir('training_results')

    conf_dir = os.path.dirname(options.config) if not PRELOAD else f'training_results/{MODEL_NAME}'
    conf_yaml = options.config if not PRELOAD else os.path.join(conf_dir, 'default_hyperion_config.yml')

    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)
        train_conf = conf['training_options']

    #if PRELOAD, load the history file to get the learning rates
    if PRELOAD:
        log.info(f'Loading pretrained model from {conf_dir} directory...')
        import numpy as np
        history_file = os.path.join(conf_dir, 'history.txt')
        _, _, learning_rates = np.loadtxt(history_file, delimiter=',', unpack=True)
        preload_lr = learning_rates[-1] if learning_rates.size > 1 else learning_rates
            

    NUM_EPOCHS            = int(train_conf['num_epochs']) if not PRELOAD else int(train_conf['num_epochs']) - learning_rates.size
    BATCH_SIZE            = int(train_conf['batch_size'])
    INITIAL_LEARNING_RATE = float(train_conf['initial_learning_rate']) if not PRELOAD else preload_lr/2

    WAVEFORM_MODEL = conf['waveform']
    PRIOR_PATH     = os.path.join(conf_dir, 'default_EFB-T_prior.yml') if PRELOAD else os.path.join(conf_dir, 'priors/'+conf['prior']+'.yml')
    DURATION       = conf['duration']
    
    if DEVICE is None:
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'
    
        
    with torch.device(DEVICE):
        """
        SETUP DETECTOR NETWORK AND ASD SAMPLERS ============================================
        """
        det_network = GWDetectorNetwork(conf['detectors'], use_torch=True, device=DEVICE)
        det_network.set_reference_time(conf['reference_gps_time'])

        asd_samplers = dict()
        for ifo in det_network.detectors:
            asd_samplers[ifo] = ASD_Sampler(ifo, 
                                            device        = DEVICE,
                                            fs            = conf['fs'],
                                            fmin          = conf['fmin'],
                                            duration      = DURATION,
                                            reference_run = conf['ASD_reference_run'])
        
        """
        SETUP WAVEFORM GENERATOR ===========================================================
        """
        with open(PRIOR_PATH, 'r') as f:
            prior_conf = yaml.safe_load(f)
            wvf_kwargs = prior_conf['waveform_kwargs'] if 'waveform_kwargs' in prior_conf else dict()
        
        waveform_generator = WaveformGenerator(WAVEFORM_MODEL, fs=conf['fs'], duration=DURATION, **wvf_kwargs)
        
        
        """
        SETUP DATASET GENERATOR ===========================================================
        """
        dataset_kwargs = {'waveform_generator'      : waveform_generator, 
                              'asd_generators'      : asd_samplers,
                              'det_network'         : det_network,
                              'device'              : DEVICE,
                              'batch_size'          : BATCH_SIZE,
                              'inference_parameters': conf['inference_parameters'],
                              'prior_filepath'      : PRIOR_PATH,
                              'n_proc'              : eval(conf['training_options']['n_proc']),
                              'use_reference_asd'   : conf['use_reference_asd'],
                              'whiten_kwargs'       : conf['training_options']['whiten_kwargs']}

        train_ds = DatasetGenerator(**dataset_kwargs,
                                    random_seed=train_conf['seeds']['train'], 
                                    num_preload=conf['training_options']['num_preload_train'])
        val_ds   = DatasetGenerator(**dataset_kwargs, 
                                    random_seed=train_conf['seeds']['val'], 
                                    num_preload=conf['training_options']['num_preload_val'])        
        
        """
        SETUP FLOW MODEL ===================================================================
        """
        #checkpoint directory
        checkpoint_dir = os.path.join('training_results', MODEL_NAME)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_filepath = os.path.join(checkpoint_dir, f'{MODEL_NAME}_flow_model.pt')

        if not PRELOAD:
            #write configuaration file to checkpoint directory
            conf_yaml_write = os.path.join(checkpoint_dir, 'hyperion_config.yml')
            with open(conf_yaml_write, 'w') as yaml_file:
                yaml.dump(conf, yaml_file)
            
            #write prior file to checkpoint directory
            conf_prior_write = os.path.join(checkpoint_dir, 'prior.yml')
            with open(conf_prior_write, 'w') as yaml_file:
                with open(PRIOR_PATH, 'r') as prior:
                    prior = yaml.safe_load(prior)
                yaml.dump(prior, yaml_file)


        #set up Flow model
        if not PRELOAD:
            prior_metadata = train_ds.prior_metadata
            flow = build_flow(prior_metadata, config_file=conf_yaml).to(DEVICE)
        else:
            flow = build_flow(checkpoint_path=checkpoint_filepath).to(DEVICE)            
            '''
            print(flow.prior_metadata)
            flow.prior_metadata['inference_parameters'] = conf['inference_parameters']
            flow.prior_metadata['parameters']['luminosity_distance'] = flow.prior_metadata['parameters'].pop('distance')
            flow.prior_metadata['means']['luminosity_distance'] = flow.prior_metadata['means'].pop('distance')
            flow.prior_metadata['stds']['luminosity_distance'] = flow.prior_metadata['stds'].pop('distance')
            '''
        #set up Optimizer and Learning rate schedulers
        optim_kwargs = {'params': [p for p in flow.parameters() if p.requires_grad], 
                        'lr': INITIAL_LEARNING_RATE}
        optimizer = get_optimizer(name=train_conf['optimizer']['algorithm'], kwargs=optim_kwargs)

        scheduler_kwargs = train_conf['lr_schedule']['kwargs']
        scheduler_kwargs.update({'optimizer':optimizer})
        scheduler = get_LR_scheduler(name = train_conf['lr_schedule']["scheduler"], 
                                        kwargs = scheduler_kwargs )
        
        #set up Trainer
        trainer_kwargs = {'optimizer'          : optimizer, 
                          'scheduler'          : scheduler,
                          'checkpoint_filepath': checkpoint_filepath,
                          'steps_per_epoch'    : train_conf['steps_per_epoch'],
                          'val_steps_per_epoch': train_conf['val_steps_per_epoch'],
                          'verbose'            : train_conf['verbose'],
                          'add_noise'          : train_conf['add_noise']
                        }

        flow_trainer = Trainer(flow, train_ds, val_ds, device=DEVICE, **trainer_kwargs)
        
        flow_trainer.train(NUM_EPOCHS, overwrite_history=False if PRELOAD else True)