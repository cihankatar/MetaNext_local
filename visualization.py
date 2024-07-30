
import numpy as np
import matplotlib.pyplot as plt

def test_image(im_test, prediction, labels):
    rand_idx = 1 #np.random.randint(batch_size)
    prediction    = prediction[1]        ## (1, 512, 512)
    prediction    = np.squeeze(prediction)     ## (512, 512)
    prediction    = prediction > 0.5
    prediction    = np.array(prediction, dtype=np.uint8)
    prediction    = np.transpose(prediction)

    im_test       = np.array(im_test[rand_idx]*255,dtype=int)
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


def supervised(images,labels):
    import matplotlib.pyplot as plt

    which_image=np.random.randint(5)
    image=images[which_image].permute(2,1,0)
    label=labels[which_image].permute(2,1,0)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image) 
    plt.subplot(1, 2, 2)
    plt.imshow(label)

    return image,label


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
