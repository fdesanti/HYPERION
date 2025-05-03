import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from ..core.flow import build_flow
from ..core.utilities import HYPERION_Logger, latexify

log = HYPERION_Logger()

class Trainer: 
    """
    Class that handles the training of a flow model.
    The training is performed over a number of epochs, where each epoch
    is defined as when the model has been optimized on a number of batches
    equal to the specified training steps.

    Args:
        flow                           (Flow): the flow model to be trained
        training_dataset   (DatasetGenerator): Training dataset
        validation_dataset (DatasetGenerator): Validation dataset
        optimizer     (torch.optim.Optimizer): Optimizer to be used
        scheduler  (torch.optim.lr_scheduler): Learning rate scheduler
        device                          (srt): Device where the model is going to be trained ('cpu' or 'cuda'). (Default: 'cpu')
        checkpoint_filepath             (str): Path to the file where the model's weights are going to be saved
        steps_per_epoch                 (int): Number of batches to be used in each epoch. If None, it is set to ``len(training_dataset)/1000``
        val_steps_per_epoch             (int): Number of batches to be used in each validation epoch. If None, it is set to ``len(validation_dataset)/1000``
        verbose                        (bool): If True, prints training and validation losses. (Default: True)
        add_noise                      (bool): If True, adds noise to the training dataset. (Default: True)
    """
    def __init__(self,
                flow,
                training_dataset,
                validation_dataset,
                optimizer          : torch.optim.Optimizer,
                scheduler          : torch.optim.lr_scheduler,
                device,
                checkpoint_filepath,
                steps_per_epoch     = None,
                val_steps_per_epoch = None,
                verbose             = True,
                add_noise           = True
                ):
        
        self.device     = device
        self.flow       = flow.float()
        self.train_ds   = training_dataset
        self.val_ds     = validation_dataset
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.add_noise  = add_noise
        
        self.checkpoint_filepath = checkpoint_filepath
        self.checkpoint_dir      = os.path.dirname(checkpoint_filepath)

        if steps_per_epoch is not None:
            self.steps_per_epoch = steps_per_epoch
        else:
            self.steps_per_epoch = len(self.train_ds) // self.train_ds.batch_size
        
        if val_steps_per_epoch is not None:
            self.val_steps_per_epoch = val_steps_per_epoch
        else:
            self.val_steps_per_epoch = len(self.val_ds) // self.val_ds.batch_size
             
        self.verbose = verbose
  

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
            parameters, strains = self.train_ds.__getitem__(add_noise=self.add_noise)
            
            #training step
            self.optimizer.zero_grad()
            
            #get the loss
            log_p = -self.flow.log_prob(parameters.to(self.device), strains.to(self.device))#, asd.to(self.device))
            loss  = torch.mean(log_p)
            
            if self.verbose:
                    print(f'Epoch = {epoch} |  Step = {step+1} / {self.steps_per_epoch}  |  Loss = {loss.item():.3f}', end='\r')
            
            if torch.isnan(loss) or torch.isinf(loss):
                #do not update model's weights
                fail_counter += 1
            else:
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
            parameters, strains = self.val_ds.__getitem__(add_noise=self.add_noise)
            
            #computing loss
            log_p = -self.flow.log_prob(parameters.to(self.device), strains.to(self.device))#, asd.to(self.device))
            loss  = torch.mean(log_p)
           
            if self.verbose:
                print(f'Epoch = {epoch}  |  Validation Step = {step+1} / {self.val_steps_per_epoch}  |  Loss = {loss.item():.3f}', end='\r')
            
            if torch.isnan(loss) or torch.isinf(loss):
                fail_counter += 1            
            else:
                avg_val_loss += loss.item() #item() returns loss as a number instead of a tensor
            #if step> self.val_steps_per_epoch:
            #    break
            #step+=1
        if fail_counter >= 0.5* self.val_steps_per_epoch:
            return np.nan
        
        avg_val_loss /= (self.val_steps_per_epoch-fail_counter)
        return avg_val_loss
  
    @staticmethod
    def save_checkpoint(flow, checkpoint_filepath, verbose=True):
        """
        Save the model's weights and optimizer's state to a file

        Args:
            flow               (Flow): the model to be saved
            checkpoint_filepath (str): path to the file where the model's weights are going to be saved
        """
        
        checkpoints = {
            'configuration': flow.configuration,
            'prior_metadata': flow.prior_metadata,
            'model_state_dict': flow.state_dict(),
            #'optimizer_state_dict': self.optimizer.state_dict(),
            #'epoch': epoch,
        }
        torch.save(checkpoints, checkpoint_filepath)
        if verbose:
            log.info(f"Training checkpoint saved at {checkpoint_filepath}")

    @staticmethod
    def load_trained_flow(checkpoint_filepath, device, flow):   
        """
        Load the trained flow from the checkpoint file. If the checkpoint file does not exist,
        it returns the flow model at the last epoch.

        Args:
            checkpoint_filepath (str): Path to the file where the model's weights are saved
            device              (str): Device where the model is going to be loaded ('cpu' or 'cuda')
            flow               (Flow): The flow model at last epoch.

        Returns:
            trained_flow (Flow): The trained flow model
        """

        if os.path.exists(checkpoint_filepath):
            try:
                best_trained_flow = build_flow(checkpoint_path=checkpoint_filepath).to(device)
                log.info(f"Model loaded from checkpoint at {checkpoint_filepath}")
            except Exception as e:
                #loading manually weights
                weights = torch.load(checkpoint_filepath, map_location=torch.device('cpu'), weights_only=False)
                flow.load_state_dict(weights['model_state_dict'])
                best_trained_flow = flow
        else:
            log.error(f"Checkpoint not found at {checkpoint_filepath}")
            log.info("Returning model at last epoch")
            best_trained_flow = flow
        return best_trained_flow
    
    @staticmethod
    @latexify
    def make_history_plots(history_filepath, savepath):
        """
        Make a plot of the training history

        Args:
            history_filepath (str): Path to the file where the history is saved
            savepath         (str): Path to the file where the plot is going to be saved
        """
        
        train_loss, val_loss, lr = np.loadtxt(history_filepath, delimiter=',',unpack = True)
        epochs = np.arange(len(train_loss))+1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # history
        ax1.plot(epochs, train_loss, label='train loss')
        ax1.plot(epochs, val_loss, label='val loss')
        ax1.legend()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('$KL[p || q]$')
        ymin = min(np.min(train_loss), np.min(val_loss)) - 0.5
        ymax = train_loss[0] + 0.5
        ax1.set_ylim(ymin, ymax)
        
        # learning_rate
        ax2.plot(epochs, lr, label='learning rate')
        ax2.legend()
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('$\eta$')
        
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        plt.close(fig)

        return fig
    
	
    def train(self, num_epochs, overwrite_history=True):
        """
        Main training loop over the epochs.
        
        Args:
            num_epochs           (int): Number of epochs to train the model
            overwrite_history   (bool): If True, the history file is overwritten. (Default: True)

        Returns:
            trained_flow (Flow): The trained Flow model

        Hint:
            Set ``overwrite_history = False`` when resuming a training.
        """
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
        log.info('Starting Training...\n')
        
        #main training loop over the epochs
        for epoch in tqdm(range(1,num_epochs+1)):

            #preload waveforms
            if hasattr(self.train_ds, 'preload_waveforms'):
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
                log.error(f'Epoch {epoch} skipped due to nan loss')
                log.info(f'Loading previously saved model\n')
                self.flow = build_flow(checkpoint_path=self.checkpoint_filepath).to(self.device)
                continue #we skip to next iteration
            
            if self.verbose:
                log.info(f'Epoch = {epoch}  |  avg train loss = {train_loss:.3f}  |  avg val loss = {val_loss:.3f}')
           
            #save updated model weights and update best values
            if (train_loss < best_train_loss) and (val_loss < best_val_loss):
                self.save_checkpoint(self.flow, self.checkpoint_filepath, self.verbose)
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
                self.make_history_plots(history_filepath=self.checkpoint_dir+'/history.txt', 
                                        savepath=self.checkpoint_dir+'/history_plot.jpg')
            except:
                pass
            
            #perform learning rate schedule step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss) #for ReduceLrOnPlateau
                updated_lr = self.scheduler.get_last_lr()[0]
                if updated_lr < current_lr:
                    log.info(f"Reduced learning rate to {updated_lr}")
            else:
                self.scheduler.step() #for CosineAnnealingLr
            
        f.close()
        log.info('Training Completed!\n')

        return self.load_trained_flow(self.checkpoint_filepath, self.device, self.flow)