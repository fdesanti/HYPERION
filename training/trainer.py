"""This file contains the Trainer class which is used to handle the
    training procedure of HYPERION
"""

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from gwskysim.gwskysim.utilities.gwlogger import GWLogger

sns.set_theme()
sns.set_context("talk")


class Trainer:
    def __init__(self,
                flow:           torch.nn.Module,
                training_dataset:   torch.utils.data.DataLoader,
                validation_dataset  :   torch.utils.data.DataLoader,
                optimizer:      torch.optim.Optimizer,
                scheduler:      torch.optim.lr_scheduler,
                device:         str,
                checkpoint_filepath: str,
               
                
                steps_per_epoch     = None,
                val_steps_per_epoch = None,
                verbose = True,
                
                ):
        
        self.device     = device
        self.flow       = flow.to(device)
        self.train_ds   = training_dataset
        self.val_ds     = validation_dataset
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        
        self.checkpoint_filepath = checkpoint_filepath
        self.checkpoint_dir      = os.path.dirname(checkpoint_filepath)
        self.steps_per_epoch     = steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch

        self.verbose = verbose
        return

    def _train_on_epoch(self, epoch):
        avg_train_loss = 0
        if self.steps_per_epoch is not None:
            total_steps = self.steps_per_epoch
        else:
            total_steps = len(self.train_ds)
        step         = 1
        fail_counter = 1

        for parameters, strains in self.train_ds:
            
            #moving batch on device
            parameters = parameters.to(self.device)
            strains    = strains.to(self.device)
            
            #training step
            self.optimizer.zero_grad()
            
            log_p =  -self.flow.log_prob(parameters, strains)
            loss  = torch.mean(log_p)
            if torch.isnan(loss) or torch.isinf(loss):
                fail_counter += 1
                step -= 1
            else:
                if self.verbose:
                    print(f'Epoch = {epoch} |  Step = {step} / {total_steps}  |  Loss = {loss.item():.3f}', end='\r')
            #print(loss)
            
                #updating weights
                loss.backward()
                avg_train_loss += loss.item() #item() returns loss as a number instead of a tensor
            
            
                
            
            self.optimizer.step()
            
            
            
            if step == total_steps:
                break
            
            step+=1
        if fail_counter >= 0.5*total_steps:
            return np.nan
       
        
        avg_train_loss /= step
    
        return avg_train_loss
   
    def _test_on_epoch(self, epoch):
        avg_val_loss = 0
        if self.val_steps_per_epoch is not None:
            total_steps = self.val_steps_per_epoch
        else:
            total_steps = len(self.val_ds)

        step         = 1
        fail_counter = 1

        for parameters, strains in self.val_ds:
            
            #moving batch on device
            parameters = parameters.to(self.device)
            strains    = strains.to(self.device)
            
            #computing loss
            log_p = - self.flow.log_prob(parameters, strains)
            loss  = torch.mean(log_p)#.detach()
            
            if torch.isnan(loss) or torch.isinf(loss):
                fail_counter += 1
                step         -= 1
            else:
         
                avg_val_loss += loss.item() #item() returns loss as a number instead of a tensor

                if self.verbose:
                    print(f'Epoch = {epoch}  |  Validation Step = {step} / {total_steps}  |  Loss = {loss.item():.3f}', end='\r')
            
                #return np.nan
            
            if step == total_steps:
                break
            step+=1
        
        if fail_counter >= 0.5* total_steps:
            return np.nan
        
        avg_val_loss /= step
        return avg_val_loss
  
    
    def _save_checkpoint(self, epoch):
        
        checkpoints = {
            'configuration': self.flow.configuration,
            'model_hyperparams': self.flow.model_hyperparams,
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
        ymin = np.min(train_loss)-0.5
        ymax = train_loss[0]+0.5
        plt.ylim(ymin, ymax)
        #learning_rate
        plt.subplot(122)
        plt.plot(epochs, lr, label = 'learning rate')
        plt.legend()
        plt.xlabel('epoch')
        plt.savefig(self.checkpoint_dir+'/history_plot.jpg', dpi=200)
        plt.close()

        #best_epoch = np.argwhere(train_loss==train_loss.min())[0][0]#best epoch for validation
        #print(f"best train loss = {train_loss[best_epoch]:.3f} at epoch {epochs[best_epoch]}")
        #best_epoch = np.argwhere(val_loss==val_loss.min())[0][0] #best epoch for validation
        #print(f"best val   loss = {val_loss[best_epoch]:.3f} at epoch {epochs[best_epoch]}\n")
        
	
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
        
       
        
        self.log.info('Starting Training...\n')
        
        #subset_imax = self.Nsubs - len(self.train_dataset)
        for epoch in tqdm(range(1,num_epochs+1)):
            #select dataset subset
            #start = np.random.randint(0, subset_imax)
            #train_indices = np.arange(start, start+self.Nsubs)
            #train_subset = Subset(self.train_dataset, train_indices)
            #self.train_ds = DataLoader(train_subset, **self.data_loader_kwargs)




            #on-epoch training
            self.flow.train(True) #train attribute comes from nn.Module and is used to set the weights in training mode
            train_loss = self._train_on_epoch(epoch)
            
            #on-epoch validation
            self.flow.eval()      #eval attribute comes from nn.Module and is used to set the weights in evaluation mode
            with torch.no_grad():
                val_loss = self._test_on_epoch(epoch)
            
            if np.isnan(train_loss) or np.isnan(val_loss):
                self.log.error(f'Epoch {epoch} skipped due to nan loss\n')
                continue #we skip to next iteration
        
            self.log.info(f'Epoch = {epoch}  |  avg train loss = {train_loss:.3f}  |  avg val loss = {val_loss:.3f}')
           
            if (train_loss < best_train_loss) and (val_loss < best_val_loss):
                self._save_checkpoint(epoch+1)
                best_train_loss = train_loss
                best_val_loss   = val_loss
                print(f"best train loss = {best_train_loss:.3f} at epoch {epoch}")
                print(f"best val   loss = {best_val_loss:.3f} at epoch {epoch}\n")
           
           
            #make lr scheduler step and get current lr
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss) #for ReduceLrOnPlateau
            else:
                lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step() #for CosineAnnealingLr
                
            
            #write history to file
            f.write(str(train_loss)+','+str(val_loss)+','+str(lr)+'\n')
            f.flush()
                
            
            
            
            try:
                self._make_history_plots()
            except:
                pass
            
        f.close()
        self.log.info('Training Completed!\n')
         


    
    
    
