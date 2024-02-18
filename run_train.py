import os
import json
import torch

from hyperion.training import *
from hyperion.config import CONF_DIR
from hyperion.core.flow import build_flow

from gwskysim.gwskysim.sources import EffectiveFlyByTemplate
from gwskysim.gwskysim.detectors import GWDetector




def run_train():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    work_dir = os.getcwd()
    parent_dir, _ = os.path.split(work_dir)
    dir_name = 'fdesanti-work1' #if mode == 'BHBH' else 'fdesanti-work1'
    work_dir = os.path.join(parent_dir, dir_name)
    
    
    dataset_dir = glob.glob(work_dir+'/GWsim_'+mode+'*')[0]

    data_manager = DatasetManager(dataset_directory=dataset_dir, train_split=0.9)

    train_files = data_manager.dataset_files
    val_files   = data_manager.dataset_files

    train_parameter_df = data_manager.train_parameter_df
    val_parameter_df   = data_manager.val_parameter_df


        
    data_loader_kwargs = {'batch_size': BATCH_SIZE, 'pin_memory':False,
                            'num_workers': 32, 'prefetch_factor': 2, 
                            'drop_last': True, 
                            'shuffle' : True}
        
        
        
        
        
    train_dataset = SimpleSingleWaveformDataset(sources_fpaths=train_files, parameters_df=train_parameter_df,
                                                  parameters_bounds=data_manager.parameters_bounds.copy())
    train_loader = DataLoader(train_dataset, 
                                **data_loader_kwargs,
                                sampler = None
                                )  
     
    val_dataset = SimpleSingleWaveformDataset(sources_fpaths=val_files, parameters_df=val_parameter_df,
                                                parameters_bounds=data_manager.parameters_bounds.copy())
    val_loader = DataLoader(val_dataset, 
                                **data_loader_kwargs,
                                sampler = None
                            )  
        
        
        
        
        
        
      #FLOW -------------------------------------------------------
    model_hyperparams = {'means': train_dataset.means, 'stds': train_dataset.stds, 
                                'parameters_names': train_dataset.out_parameters_names,
                                'parameters_bounds': train_dataset.parameters_bounds}
    
    
    if load_trained:    
        checkpoint_path = 'training_results/BEST_BHBH/BEST_BHBH_flow_model.pt'
        flow = build_flow(checkpoint_path=checkpoint_path)
        flow.model_hyperparams=model_hyperparams
        print('----> loaded pre-trained model')
    else:
        flow = build_flow(model_hyperparams)
        
      

    optimizer   = Adam([p for p in flow.parameters() if p.requires_grad], lr = INITIAL_LEARNING_RATE) 
      #scheduler   = CosineAnnealingLR(optimizer, NUM_EPOCHS, 0)
    scheduler   = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 15, mode = 'min', threshold = 0, verbose = True)
        #scheduler = StepLR(optimizer, step_size=2, gamma = 0.5)
        #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 50, T_mult=2)
        
    output_model_dir = 'training_results/'+mode
    if not os.path.exists(output_model_dir):
        os.mkdir(output_model_dir)
            
    checkpoint_filepath = os.path.join(output_model_dir, mode+'_flow_model.pt')

    trainer_kwargs = {'optimizer': optimizer, 'scheduler':scheduler, 
                        'checkpoint_filepath': checkpoint_filepath,
                        'steps_per_epoch' : 1000,
                        'val_steps_per_epoch' : 250,
                        'verbose': True, 
                        'data_loader_kwargs': data_loader_kwargs,
                        #'train_dataset': train_dataset
                        }


        
    flow_trainer = Trainer(flow, train_loader, val_loader, device=device, **trainer_kwargs)
        
    flow_trainer.train(NUM_EPOCHS)

    train_dataset.close_hf()
    val_dataset.close_hf()





if __name__ == '__main__':

    conf_json = CONF_DIR + '/train_config.json'
    
    with open(conf_json) as json_file:
        conf = json.load(json_file)

    
    NUM_EPOCHS = int(conf['num_epochs'])
    BATCH_SIZE = int(conf['batch_size'])
    INITIAL_LEARNING_RATE = float(conf['initial_learning_rate'])

    device = conf['device']
    
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
                          'device':device, 'batch_size': BATCH_SIZE}

        train_ds = DatasetGenerator(**dataset_kwargs, random_seed=conf['train_seed'])
        val_ds = DatasetGenerator(**dataset_kwargs, random_seed=conf['train_seed'])        
        
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

        checkpoint_filepath = os.path.join('training_results', 'BHBH_flow_model.pt')

        trainer_kwargs = {'optimizer': optimizer, 'scheduler':scheduler, 
                        'checkpoint_filepath': checkpoint_filepath,
                        'steps_per_epoch' : conf['steps_per_epoch'],
                        'val_steps_per_epoch' : conf['val_steps_per_epoch'],
                        'verbose': conf['verbose'], 
                        }

        flow_trainer = Trainer(flow, train_ds, val_ds, device=device, **trainer_kwargs)
        
        flow_trainer.train(NUM_EPOCHS)
        






    







    #run_train()


