"""This file contains the Trainer class which is used to handle the
    training procedure of HYPERION
"""

import os
import torch
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from gwskysim.gwskysim.utilities.gwlogger import GWLogger

from ..core.flow import build_flow


class Trainer:
    def __init__(self,
                flow: torch.nn.Module,
                training_dataset: torch.utils.data.DataLoader,
                validation_dataset: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                device: str,
                checkpoint_filepath: str,
                steps_per_epoch     = None,
                val_steps_per_epoch = None,
                verbose = True,
                ):
        
        self.device     = device
        self.flow       = flow.float()
        self.train_ds   = training_dataset
        self.val_ds     = validation_dataset
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        
        self.checkpoint_filepath = checkpoint_filepath
        self.checkpoint_dir      = os.path.dirname(checkpoint_filepath)

        if steps_per_epoch is not None:
            self.steps_per_epoch = steps_per_epoch
        else:
            self.steps_per_epoch = len(self.train_ds) // 1000
        
        if val_steps_per_epoch is not None:
            self.val_steps_per_epoch = val_steps_per_epoch
        else:
            self.val_steps_per_epoch = len(self.val_ds) // 1000
             
        self.verbose = verbose
        return

    def _train_on_epoch(self, epoch):
        """
        Training over one epoch where an epoch is defined as when
        the model has been optimized on a number of batches equal
        to the specified training steps
        """

        avg_train_loss = 0
        fail_counter = 0
        #step = 0
        #main loop over the epoch's batches
        for step in range(self.steps_per_epoch):
        #for parameters, strains in self.train_ds:
            #getting the trainig batch
            parameters, strains, asd = self.train_ds.__getitem__()
            
            
            #training step
            self.optimizer.zero_grad()
            
            #get the loss
            log_p =  -self.flow.log_prob(parameters.to(self.device), strains.to(self.device), asd.to(self.device))
            loss  = torch.mean(log_p)
            
            if torch.isnan(loss) or torch.isinf(loss):
                #do not update model's weights
                fail_counter += 1
            else:
                if self.verbose:
                    print(f'Epoch = {epoch} |  Step = {step+1} / {self.steps_per_epoch}  |  Loss = {loss.item():.3f}', end='\r')
            
                #updating weights
                loss.backward()
                avg_train_loss += loss.item() #item() returns loss as a number instead of a tensor
            
            #perform the single step gradient descend
            self.optimizer.step()
            #if step > self.steps_per_epoch:
            #    break
            #step+=1
            

        if fail_counter >= 0.5*self.steps_per_epoch:
            #something went wrong too many times during the epoch
            #better to leave the model as it is
            return np.nan
       
        #compute the mean loss
        avg_train_loss /= (self.steps_per_epoch-fail_counter)
    
        return avg_train_loss
   

    def _test_on_epoch(self, epoch):
        """
        Validation over one epoch where an epoch is defined as when
        the model has been optimized on a number of batches equal
        to the specified training steps
        """

        avg_val_loss = 0
        fail_counter = 0
        #step=0
        for step in range(self.val_steps_per_epoch):
        #for parameters, strains in self.val_ds:
            
            #getting batch
            parameters, strains, asd = self.val_ds.__getitem__()
            
            #computing loss
            log_p = -self.flow.log_prob(parameters.to(self.device), strains.to(self.device), asd.to(self.device))
            loss  = torch.mean(log_p)
           
            if torch.isnan(loss) or torch.isinf(loss):
                fail_counter += 1
                print(f'Epoch = {epoch}  |  Validation Step = {step+1} / {self.val_steps_per_epoch}  |  Loss = {loss.item():.3f}', end='\r')
            
            else:
                avg_val_loss += loss.item() #item() returns loss as a number instead of a tensor
                if self.verbose:
                    print(f'Epoch = {epoch}  |  Validation Step = {step+1} / {self.val_steps_per_epoch}  |  Loss = {loss.item():.3f}', end='\r')
            #if step> self.val_steps_per_epoch:
            #    break
            #step+=1
        if fail_counter >= 0.5* self.val_steps_per_epoch:
            return np.nan
        
        avg_val_loss /= (self.val_steps_per_epoch-fail_counter)
        return avg_val_loss
  
    
    def _save_checkpoint(self, epoch):
        
        checkpoints = {
            'configuration': self.flow.configuration,
            'prior_metadata': self.flow.prior_metadata,
            'model_state_dict': self.flow.state_dict(),
            #'optimizer_state_dict': self.optimizer.state_dict(),
            #'epoch': epoch,
        }
            
        torch.save(checkpoints, self.checkpoint_filepath)
        if self.verbose:
            self.log.info(f"Training checkpoint saved at {self.checkpoint_filepath}")
    
    def _make_history_plots(self):
        train_loss, val_loss, lr = np.loadtxt(self.checkpoint_dir+'/history.txt', delimiter=',',unpack = True)
        epochs = np.arange(len(train_loss))+1

        plt.figure(figsize=(20, 8))
        #history
        plt.subplot(121)
        plt.plot(epochs, train_loss, label ='train loss')
        plt.plot(epochs, val_loss, label = 'val loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('$KL[p || q]$')
        ymin = min(np.min(train_loss), np.min(val_loss))-0.5
        ymax = train_loss[0]+0.5
        plt.ylim(ymin, ymax)
        
        #learning_rate
        plt.subplot(122)
        plt.plot(epochs, lr, label = 'learning rate')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('$\eta$')
        plt.savefig(self.checkpoint_dir+'/history_plot.jpg', dpi=200, bbox_inches='tight')
        plt.close()
    
	
    def train(self, num_epochs, overwrite_history=True):
        self.log = GWLogger('training_logger')
        self.log.setLevel('INFO')

        best_train_loss = np.inf
        best_val_loss   = np.inf
        

        self.history_fpath = os.path.join(self.checkpoint_dir, 'history.txt')
        
        if not overwrite_history:
            f = open(self.history_fpath, 'a')
        else:
            f = open(self.history_fpath, 'w')
            f.write('#training loss, validation loss, learning rate\n')
            f.flush()
        
        print('\n')
        self.log.info('Starting Training...\n')
        
        #main training loop over the epochs
        for epoch in tqdm(range(1,num_epochs+1)):

            #preload waveforms
            self.train_ds.preload_waveforms()
            self.val_ds.preload_waveforms()

            
            #on-epoch training
            self.flow.train(True) #train attribute comes from nn.Module and is used to set the weights in training mode
            train_loss = self._train_on_epoch(epoch)
            
            #on-epoch validation
            self.flow.eval()      #eval attribute comes from nn.Module and is used to set the weights in evaluation mode
            with torch.inference_mode():
                val_loss = self._test_on_epoch(epoch)
            
            if np.isnan(train_loss) or np.isnan(val_loss):
                self.log.error(f'Epoch {epoch} skipped due to nan loss')
                self.log.info(f'Loading previously saved model\n')
                self.flow = build_flow(checkpoint_path=self.checkpoint_filepath).to(self.device)
                continue #we skip to next iteration
        
            self.log.info(f'Epoch = {epoch}  |  avg train loss = {train_loss:.3f}  |  avg val loss = {val_loss:.3f}')
           
            #save updated model weights and update best values
            if (train_loss < best_train_loss) and (val_loss < best_val_loss):
                self._save_checkpoint(epoch+1)
                best_train_loss = train_loss
                best_val_loss   = val_loss
                print(f"best train loss = {best_train_loss:.3f} at epoch {epoch}")
                print(f"best val   loss = {best_val_loss:.3f} at epoch {epoch}\n")
           
            
            #get current learning rate
            if epoch > 1:
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
            
            #write history to file
            f.write(str(train_loss)+','+str(val_loss)+','+str(current_lr)+'\n')
            f.flush()
            
            #make history plot
            try:
                self._make_history_plots()
            except:
                pass
            
        
            #perform learning rate schedule step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss) #for ReduceLrOnPlateau
                updated_lr = self.scheduler.get_last_lr()[0]
                if updated_lr < current_lr:
                    self.log.info(f"Reduced learning rate to {updated_lr}")
            else:
                self.scheduler.step() #for CosineAnnealingLr
                
            

        f.close()
        self.log.info('Training Completed!\n')
         


    
    
    
