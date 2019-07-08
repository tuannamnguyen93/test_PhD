import numpy as np
import matplotlib.pyplot as plt
import os
print(len("plot_info"))
for file in os.listdir("plot_info"):
    
    name=file.split("_")
    acc=np.load("plot_info/"+file)
    color=""
    
    if name[0] == "googlenet":
        color += 'r'
    elif name[0] == "resnet":
        color += "b"
    elif name[0] == "vgg":
        color += "g"
    elif name[0] == "alexnet":
        color += "y"
        
    if name[2] == "Imagenet":
        color+= "-"
    elif name[2] == "no" :
        color+= "--"
    elif name[2] == "Document":
        color+= "-."
    print (name)
    print(acc)
    plt.plot(acc[:,1],acc[:,0],color,label=name[0]+"_"+name[2])
plt.ylabel('mean_accuracy')
plt.xlabel('number_of_sample')
plt.axis([20,160, 0, 1])
plt.legend()
plt.savefig('result.png')
