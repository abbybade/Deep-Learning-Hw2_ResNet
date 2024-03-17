import numpy as np
import matplotlib.pyplot as plt  
def plot_losses_epochs(losses=[], epochs=[]):
    
    # Create the plot
    plt.figure(figsize=(8, 6)) 
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b') 
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs' )
    plt.grid(True)
    plt.legend()  
    plt.savefig(fname="ResNet")
    plt.close()
        