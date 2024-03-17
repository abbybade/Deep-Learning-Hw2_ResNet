import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,

            ### LR
            ### OPTIMIZER
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.network.parameters(), lr=self.config.learning_rate,
                                   momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        # If you want to use Adam, replace the line above with the following one:
        # self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate,
        #                             weight_decay=self.config.weight_decay)
        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        initial_lr = 0.1
        losses = []
        epoches = []
        # Epochs after which to reduce learning rate
        lr_decay_epochs = [50, 100] 
        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            if epoch in lr_decay_epochs:
                new_lr = initial_lr / 10.0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                start_idx = i * self.config.batch_size
                end_idx = min((i + 1) * self.config.batch_size, num_samples)
                batch_x = curr_x_train[start_idx:end_idx]
                batch_y = curr_y_train[start_idx:end_idx]
                # Don't forget to use "parse_record" to perform data preprocessing.
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                processed_images = []
                for record in batch_x:
                    processed_image = parse_record(record, training=True)
                    processed_images.append(processed_image)

                # Stack the processed images back into a batch
                processed_batch_x = np.stack(processed_images)#
                
                processed_batch_x_tensor = torch.tensor(processed_batch_x, dtype=torch.float32).to(device)#
                

                # Don't forget L2 weight decay
                l2_reg = torch.tensor(0.0).to(device)
                for param in self.network.parameters():
                    l2_reg += torch.norm(param , p=2).to(device)

                # Forward pass
                processed_batch_x_tensor = processed_batch_x_tensor

                outputs = self.network(processed_batch_x_tensor)
                
                # Calculate loss
                batch_y = torch.tensor(batch_y, dtype=torch.long).to(device)
                loss = self.criterion(outputs, batch_y)
                ### YOUR CODE HERE
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                #print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True);
                
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration));
            losses.append(float(loss))
            epoches.append(int(epoch))
            
            if epoch % self.config.save_interval == 0:
                self.save(epoch)
            
        return losses, epoches

    def test_or_validate(self, x, y, checkpoint_num_list):
        self.network.eval()
        print('### Test or Validation ###')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)

            preds = []
            y = torch.tensor(y, dtype=torch.long).to(device)
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                processed_x = parse_record(x[i], training=False).reshape(1, 3, 32, 32)
                # Forward pass
                outputs = self.network(torch.tensor(processed_x).to(device))

                # Get predicted class
                _, predicted = torch.max(outputs, 1)
                preds.append(predicted.item())
                    ### END CODE HERE

            
            preds = torch.tensor(preds).to(device)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
        
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))