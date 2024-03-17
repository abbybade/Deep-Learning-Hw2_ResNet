from ImageUtils import parse_record
import matplotlib.pyplot as plt
from DataReader import load_data, train_valid_split
from Model import Cifar
import torch
import itertools
import subprocess
import os
import argparse


def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE
    parser.add_argument("--resnet_version", type=int, default=1, help="the version of ResNet")
    parser.add_argument("--resnet_size", type=int, default=3, 
                        help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--first_num_filters", type=int, default=32, help='first filter in the stack')#
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
    ### YOUR CODE HERE
    parser.add_argument("--learning_rate", type=float, default=0.1, help='initial learning rate')
    parser.add_argument("--momentum", type=float, default=0.9, help='momentum for the optimizer')
    
      #parameters to test 
        #resnet_versions = [1, 2]
        #resnet_sizes = [3, 5, 7]
        #batch_sizes = [64, 128, 256]
        #learning_rates = [0.001, 0.01, 0.1]
        #momentums = [0.8, 0.9, 0.95]
        #weight_decays = [1e-4, 2e-4, 5e-4]
        #first_num_filters = [16, 32, 64]
    return parser.parse_args()

def main(config):

    print("--- Preparing Data ---")
    data_dir = "cifar-10-python.tar"
    x_train, y_train, x_test, y_test = load_data(data_dir) 
    x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cifar(config).to(device)
    
    # Use the train_new and valid set for hyperparameter tuning
    # Assuming model.train() and model.test_or_validate() are defined to handle the data accordingly
    
    print("---Training and Hyperparameter Validation ---")
    model.train(x_train_new, y_train_new, 150)  # Adjust epochs
    
    #checkpoints = [160, 170, 180, 190]  
    model.test_or_validate(x_valid, y_valid,[160,170,180,190])
    
    #validation_losses = []
    #checkpoints = [160, 170, 180, 190]  
  
    #for checkpoint in checkpoints:
        #model.test_or_validate(x_valid, y_valid, [checkpoint])
       # loss = model.test_or_validate(x_valid, y_valid, [checkpoint])
       # validation_losses.append(loss)
        
    # Plot validation losses
    #plt.figure(figsize=(10, 6))
    #plt.plot(checkpoints, validation_losses, marker='o', linestyle='-', color='blue')
    #plt.title('Validation Loss Across Epochs')
    #plt.xlabel('Epoch')
    #plt.ylabel('Validation Loss')
    #plt.xticks(checkpoints)
    #plt.grid(True)
    #plt.savefig('validation_loss_vs_epoch.png')
    #plt.show()
    
    
    print("--- Final Training & Testing ---")
    
    model.train(x_train, y_train, 10)  # Full training
    final_test_accuracy = model.test_or_validate(x_test, y_test, [10])  # Final testing
    print(f"Final Test Accuracy: {final_test_accuracy}")
        

   

    # Second step: with hyperparameters determined in the first run, re-train
    # your model on the original train set
    #print("--- Full Training ---")
    #model.train(x_train, y_train, 200)

    # Third step: after re-training, test your model on the test set.
    # Report testing accuracy in your hard-copy report
    #print("--- Final Testing ---")
    #model.test_or_validate(x_test, y_test, [200])
    ### END CODE HERE

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)