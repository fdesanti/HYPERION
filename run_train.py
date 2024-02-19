import os
import json
import torch
import seaborn as sns

from hyperion.training import *
from hyperion.config import CONF_DIR
from hyperion.core.flow import build_flow

from gwskysim.gwskysim.sources import EffectiveFlyByTemplate
from gwskysim.gwskysim.detectors import GWDetector

sns.set_theme()
sns.set_context("talk")


if __name__ == '__main__':

    conf_json = CONF_DIR + '/train_config.json'
    
    with open(conf_json) as json_file:
        conf = json.load(json_file)

    
    NUM_EPOCHS = int(conf['num_epochs'])
    BATCH_SIZE = int(conf['batch_size'])
    INITIAL_LEARNING_RATE = float(conf['initial_learning_rate'])

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with torch.device(device):

        #set up gwskysim detectors and asd_samplers
        detectors = dict()
        asd_samplers = dict()
        for ifo in conf['detectors']:
            detectors[ifo] = GWDetector(ifo, use_torch=True, device = device)
            asd_samplers[ifo] = ASD_sampler(ifo, device=device, fs=conf['fs'])
        
        #set up EFB-T waveform
        efbt = EffectiveFlyByTemplate(duration = 1, torch_compile=False, detectors=detectors, fs=conf['fs'], device = device)

        #setup dataset generator
        dataset_kwargs = {'waveform_generator': efbt, 'asd_generators':asd_samplers, 
                          'device':device, 'batch_size': BATCH_SIZE, 
                          'inference_parameters': conf['inference_parameters']}

        train_ds = DatasetGenerator(**dataset_kwargs, random_seed=conf['train_seed'])
        val_ds = DatasetGenerator(**dataset_kwargs, random_seed=conf['val_seed'])        
        
        #set up Flow model
        prior_metadata = train_ds._prior_metadata
        flow = build_flow(prior_metadata).to(device)

        #set up Optimizer and Learning rate schedulers
        optim_kwargs = {'params': [p for p in flow.parameters() if p.requires_grad], 
                        'lr': INITIAL_LEARNING_RATE}
        optimizer = get_optimizer(name=conf['optimizer']['algorithm'], kwargs=optim_kwargs)

        scheduler_kwargs = conf['lr_schedule']['kwargs']
        scheduler_kwargs.update({'optimizer':optimizer})
        scheduler = get_LR_scheduler(name = conf['lr_schedule']["scheduler"], 
                                     kwargs = scheduler_kwargs )
        
        #set up Trainer
        checkpoint_dir = os.path.join('training_results', 'BHBH')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_filepath = os.path.join(checkpoint_dir, 'BHBH_flow_model.pt')

        trainer_kwargs = {'optimizer': optimizer, 'scheduler':scheduler, 
                        'checkpoint_filepath': checkpoint_filepath,
                        'steps_per_epoch' : conf['steps_per_epoch'],
                        'val_steps_per_epoch' : conf['val_steps_per_epoch'],
                        'verbose': conf['verbose'], 
                        }

        flow_trainer = Trainer(flow, train_ds, val_ds, device=device, **trainer_kwargs)
        
        flow_trainer.train(NUM_EPOCHS)
        



