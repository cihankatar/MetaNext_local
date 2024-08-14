import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def testsample(images, labels, prediction,n):
    rand_idx = n #np.random.randint(batch_size)
    prediction    = prediction[rand_idx]        ## (1, 512, 512)
    prediction    = np.squeeze(prediction)     ## (512, 512)
    # prediction    = prediction > 0.5
    # prediction    = np.array(prediction, dtype=np.uint8)
    prediction    = np.transpose(prediction)

    im_test       = np.array(images[rand_idx]*255,dtype=int)
    im_test       = np.transpose(im_test, (2,1,0))
    im_test       = np.squeeze(im_test)     ## (512, 512)

    label_test    = np.array(labels[rand_idx]*255,dtype=int)
    label_test    = np.transpose(label_test)
    label_test    = np.squeeze(label_test)     ## (512, 512)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('test_image')
    plt.imshow(im_test) 
    plt.subplot(1, 3, 2)
    plt.title('prediction')
    plt.imshow(prediction)
    plt.subplot(1, 3, 3)
    plt.title('label')
    plt.imshow(label_test)


def trainsample(images,labels,model_output,n):
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt


    sigm=nn.Sigmoid() 

    which_image=n
    image       = images[which_image].permute(2,1,0)
    label       = labels[which_image].permute(2,1,0)
    prediction  = model_output[which_image].permute(2,1,0)
    prediction  = sigm(prediction).detach().numpy()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('test_image')
    plt.imshow(image) 
    plt.title('mask')
    plt.subplot(1, 3, 2)
    plt.imshow(label)
    plt.title('prediction')
    plt.subplot(1, 3, 3)
    plt.imshow(prediction)

def barcod(mask,maskh,points_p,predict,predicth,points_m,n):
    
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.title('Mask Image',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.imshow(mask.detach().numpy(),cmap='gray') 


    plt.subplot(3, 2, 2)
    plt.title('Model Prediction',fontsize = 8)
    plt.imshow(predict.detach().numpy(),cmap='gray')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)

    plt.subplot(3, 2, 3)
    plt.scatter(points_p[:,0],points_p[:,1] ,s=1)
    plt.title('Extracted Points of Mask Image',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('scaled')

    plt.subplot(3, 2, 4)
    plt.scatter(points_m[:,0],points_m[:,1] ,s=1)
    plt.title('Extracted Points of Predicted Image',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('scaled')

    # homology
    if maskh[n][1].max()>predicth[n][1].max():
        x = [0, maskh[n][1].max()]
    else:
        x = [0, predicth[n][1].max()]

    # Plot the diagonal
    mask     = maskh[n][1].cpu().detach().numpy()
    predict  = predicth[n][1].cpu().detach().numpy()

    plt.subplot(3, 2, 5)
    plt.scatter(mask[:,0],mask[:,1] ,marker='o',color='black')
    plt.plot(x, x, label='Diagonal',color='red')
    plt.title('Mask Image PH',fontsize = 8)
    plt.xlabel('Birth',fontsize = 8)
    plt.ylabel('Death',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('scaled')

    plt.subplot(3, 2, 6)
    plt.scatter(predict[:,0], predict[:,1],marker='o',color='black')
    plt.plot(x, x, label='Diagonal',color='red')
    plt.title('Prediction Image PH',fontsize = 8)
    plt.xlabel('Birth',fontsize = 8)
    plt.ylabel('Death',fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8,rotation=90)
    plt.axis('scaled')

    plt.tight_layout()

def simclr(images,labels):
    import matplotlib.pyplot as plt
#   which_batch=np.random.randint(2)
#   which_image=np.random.randint(7)

    which_batch0=0
    which_batch1=1
    image1=images[which_batch0][1].permute(2,1,0)
    image2=images[which_batch0][2].permute(2,1,0)
    image3=images[which_batch0][3].permute(2,1,0)
    image4=images[which_batch0][4].permute(2,1,0)

    label1=labels[which_batch0][1].permute(2,1,0)
    label2=labels[which_batch0][2].permute(2,1,0)
    label3=labels[which_batch0][3].permute(2,1,0)
    label4=labels[which_batch0][4].permute(2,1,0)

    image5=images[which_batch1][1].permute(2,1,0)
    image6=images[which_batch1][2].permute(2,1,0)
    image7=images[which_batch1][3].permute(2,1,0)
    image8=images[which_batch1][4].permute(2,1,0)

    label5=labels[which_batch1][1].permute(2,1,0)
    label6=labels[which_batch1][2].permute(2,1,0)
    label7=labels[which_batch1][3].permute(2,1,0)
    label8=labels[which_batch1][4].permute(2,1,0)

    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(image1)
    plt.subplot(2,4,2)
    plt.imshow(image2)
    plt.subplot(2,4,3)
    plt.imshow(image3)
    plt.subplot(2,4,4)
    plt.imshow(image4)
    
    plt.subplot(2,4,5)
    plt.imshow(label1)
    plt.subplot(2,4,6)
    plt.imshow(label2)
    plt.subplot(2,4,7)
    plt.imshow(label3)
    plt.subplot(2,4,8)
    plt.imshow(label4)

    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(image5)
    plt.subplot(2,4,2)
    plt.imshow(image6)
    plt.subplot(2,4,3)
    plt.imshow(image7)
    plt.subplot(2,4,4)
    plt.imshow(image8)
    
    plt.subplot(2,4,5)
    plt.imshow(label5)
    plt.subplot(2,4,6)
    plt.imshow(label6)
    plt.subplot(2,4,7)
    plt.imshow(label7)
    plt.subplot(2,4,8)
    plt.imshow(label8)
