import os
import yaml
import torch
import seaborn as sns

sns.set_theme()
sns.set_context("talk")

from optparse import OptionParser

from hyperion.training import *
from hyperion.config import CONF_DIR
from hyperion.core.flow import build_flow
from hyperion.simulations import (ASD_Sampler, 
                                  GWDetectorNetwork, 
                                  WaveformGenerator)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--preload", default=False, action="store_true", help="Load a pretrained model")
    (options, args) = parser.parse_args()
    
    PRELOAD = options.preload

    conf_yaml = CONF_DIR + '/hyperion_config.yml'
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)
        train_conf = conf['training_options']

    
    NUM_EPOCHS = int(train_conf['num_epochs'])
    BATCH_SIZE = int(train_conf['batch_size'])
    INITIAL_LEARNING_RATE = float(train_conf['initial_learning_rate'])

    WAVEFORM_MODEL = conf['waveform_model']
    PRIOR_PATH = os.path.join(CONF_DIR, conf['prior']+'.yml')
    DURATION  = conf['duration']
    

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = f'cuda:{num_gpus-1}'
    else:
        device = 'cpu'
    device = 'cuda'
    
    with torch.device(device):
        """
        SETUP DETECTOR NETWORK AND ASD SAMPLERS ============================================
        """
        det_network = GWDetectorNetwork(conf['detectors'], use_torch=True, device=device)
        det_network.set_reference_time(conf['reference_gps_time'])

        asd_samplers = dict()
        for ifo in det_network.detectors:
            asd_samplers[ifo] = ASD_Sampler(ifo, 
                                            device=device, 
                                            fs=conf['fs'], 
                                            duration=DURATION,
                                            reference_run=conf['ASD_reference_run'])
        
        """
        SETUP WAVEFORM GENERATOR ===========================================================
        """
        with open(PRIOR_PATH, 'r') as f:
            prior_conf = yaml.safe_load(f)
            wvf_kwargs = prior_conf['waveform_kwargs']
        
        waveform_generator = WaveformGenerator(WAVEFORM_MODEL, fs=conf['fs'], duration=DURATION, **wvf_kwargs)
        
        
        """
        SETUP DATASET GENERATOR ===========================================================
        """
        dataset_kwargs = {'waveform_generator': waveform_generator, 
                            'asd_generators':asd_samplers, 
                            'det_network': det_network,
                            #'num_preload': conf['training_options']['num_preload'],
                            'device':device, 
                            'batch_size': BATCH_SIZE, 
                            'inference_parameters': conf['inference_parameters'],
                            'prior_filepath': PRIOR_PATH, 
                            'n_proc': eval(conf['training_options']['n_proc']),
                            'use_reference_asd': conf['use_reference_asd']}

        train_ds = DatasetGenerator(**dataset_kwargs,
                                    random_seed=train_conf['seeds']['train'], 
                                    num_preload=conf['training_options']['num_preload_train'])
        val_ds   = DatasetGenerator(**dataset_kwargs, 
                                    random_seed=train_conf['seeds']['val'], 
                                    num_preload=conf['training_options']['num_preload_val'])        
        
        #checkpoint directory
        checkpoint_dir = os.path.join('training_results', 'BHBH')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_filepath = os.path.join(checkpoint_dir, 'BHBH_flow_model.pt')

        #set up Flow model
        if not PRELOAD:
            prior_metadata = train_ds.prior_metadata
            flow = build_flow(prior_metadata).to(device)
        else:
            flow = build_flow(checkpoint_path=checkpoint_filepath).to(device)            

        #set up Optimizer and Learning rate schedulers
        optim_kwargs = {'params': [p for p in flow.parameters() if p.requires_grad], 
                        'lr': INITIAL_LEARNING_RATE}
        optimizer = get_optimizer(name=train_conf['optimizer']['algorithm'], kwargs=optim_kwargs)

        scheduler_kwargs = train_conf['lr_schedule']['kwargs']
        scheduler_kwargs.update({'optimizer':optimizer})
        scheduler = get_LR_scheduler(name = train_conf['lr_schedule']["scheduler"], 
                                        kwargs = scheduler_kwargs )
        
        #set up Trainer
        trainer_kwargs = {'optimizer': optimizer, 
                            'scheduler':scheduler, 
                        'checkpoint_filepath':  checkpoint_filepath,
                        'steps_per_epoch':      train_conf['steps_per_epoch'],
                        'val_steps_per_epoch' : train_conf['val_steps_per_epoch'],
                        'verbose':              train_conf['verbose'], 
                        }

        flow_trainer = Trainer(flow, train_ds, val_ds, device=device, **trainer_kwargs)
        
        flow_trainer.train(NUM_EPOCHS, overwrite_history=False if PRELOAD else True)
        