**Behavioral Cloning**
-----------------------

The goal of the project was to train a CNN to provide steering angles based for a simulated driving on a virtual course.

### Working video
Below is a link to a video where I was able to run the model on the simulation
[![Everything Is AWESOME](https://img.youtube.com/vi/ok1UkXxivTo/0.jpg)](https://www.youtube.com/watch?v=ok1UkXxivTo)

---------------
### How to Run

#### Running Simuation ####
`python3 drive.py model.json`

Once you bring up the simulation it should connect and start providing the steering wheel angles

#### Building Model ####

###### For training the model you can run the below command ######

**`python3 model.py True .0001 'data/' 'model_base'`**

    **model.py inputs**
    1. baseModel(string)
        Need to pass what mode the model is running as, if true it will create a base model, with the name model_base.json and model_base.h5 for weights. If false it will use the model_base (.json and .h5) as a starting point and build refined models.

    2. learningRate(float)
        learning rate you want to run the model with

    3. dataLocation(string)
        Location of training data.

    4. saveModelFName(string)
        Name with which you want to save the new model. When I was running intermediate models, I named them "model_refine" and when I was satisfied I named them "model"

###### For refining an existing model ######
**`python3 model.py False .00001 'data/' 'model_refine_1'`**

You can run something like above, sending **false** will ensure the model reads _model_base_

#### Training Data ####
At first I tried using my data built using my keyboard and it really did not work. I ended up using udacity's data which was much better. The driving_log from udacity had 8036 line. Each line consisted of
* center image
* left image
* right image
* angle for steering wheel
* throttle
* break
* speed

The udacity data worked much better, as I have understood coz it used a joy stick to simulate driving which provided smoother angles. For this excercise I ignored throttle values etc and only worked with image information and angle. I read each image and the angle associated with it as a different feature and ended up with - 24108 features from the file.

###### Image Manipulation

Did two things with images:
1. Flipped them horizontally
2. Resize them to 80 rows and 160 columns. Initially I was reading the image with its existing size and that caused
a lot of memory issues for me, so I reduced it to half.

###### Augmenting data and creating more of it
Here are the ways I augmented the data
* I took the images and since we only had the angles associated with the center image, I used that angle and adjusted it for left and right images by a factor of .75 like so
    ```
    float(angle) + abs(float(angle)*.75 (For left image)
    float(angle) - abs(float(angle)*.75 (For right image)
    ```
* I also randomly(50% probability) flipped all images horizontally to give me more data. Every flipped meant I multiplied the existing angle with -1
* I removed 80% (if biasedCoin(.8):) of the images where the absolute value of the angle was less than 0.01. There were many images with small angles and 0.0 angles for the steering wheel and they made the car go straight. To balance this out I felt leaving only a few images with smaller angles was the way to go.

_Refinement - Adding more data_
Some other data augmentation techniques which i tried when I attempted to refine an existing model ref:model architecture were (these did not make it into the final model)
* Take all images that had a non zero angle for steering wheel and flip them and add the flipped image to the dataset.
* Take all images with large angles for steering wheels (0.40<abs(angle)<0.70) and flip these images and the flipped image to the dataset. I did this because in my earlier attempts the car tended to go straight and I thought by having more images with larger angles I could train the model to provide more variations.

The above data augmentation techniques where not really used in the final model as I explain below.

###### Train/Test/Validate
I wrote a custom train_test_validate method that took the generated data and split it up. Below is an example of a split from the working model run
```
Number of features after flipping images randomly - 19887
X_train has 14915 elements. removed 0
X_valid has 2485 elements. removed 0
X_test has 2487 elements. remove 0
.........
```

#### Model Architecture - Thought Process ####
The architecture was inspired by the code from comma.ai and NVIDIA ["End to End Learning for Self-Driving Cars"](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The NVIDIA paper was awesome, but there are still many things I need to understand in this paper but it gave me a great start along with code from comma.ai. I also have to thank Paul Hearty for writing this awesome guide in the forums, especially the suggestion to use generators: https://carnd-udacity.atlassian.net/wiki/questions/26214464/answers/28154732?flashId=4211a503398ab1b03a58341e8e09006cb63f82d8. When I first started I was loading all the images in memory and soon things started to fail and run out of memory. So at first I tried and reduce the image size in half, which did not help either. Finally I went down the path of the generator

_Image Generator to the rescue_
This was truly the saving grace. The image generator takes the images from the shuffled up data (containing augmented images) and processes them before feeding it to the model

_All variations and no success_
I tried different variations, from using comma.ai's model, to trying out nvidia. The advice I got from slack and forums was to train on larger angles first then gradually accept other angles and that really helped. I tried successive refinement of a 'good' model, something that could get past the bridge. At first the comma.ai model worked and I thought I was getting somewhere, but I could only get as far as after the bridge. I used that as the base model and tried various combinations (Section "Refinement - Adding more data") to refine this model. I tried lowering the learning rate, changing the angle adjustment for left and right images etc without success. I tried to document all the steps in previous_steps.txt in the same folder as README with this submission. The furthest I got was right before the final turn. Models that didn't work can be found in *attempted_models* in this submission. I only included a few, but there were way many more.

###### Finally Working Model
At that point and spending hours and hours staring at my code, I sort of took a step back and rethought the model and I settled with 4 convolutions, interspersed with dropouts, max pooling, a falttened layer, 3 hidden layers and an output layer. I feel this is a complex model but it works. I would like to simplify it, especially like the comma.ai model.

The model summary is below
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 80, 160, 3)    0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 78, 24)    1824        lambda_1[0][0]
____________________________________________________________________________________________________
leaky relu 1 (LeakyReLU)         (None, 38, 78, 24)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 37, 36)    21636       leaky relu 1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 17, 37, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
leaky relu 2 (LeakyReLU)         (None, 17, 37, 36)    0           dropout_1[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 17, 48)     43248       leaky relu 2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 7, 17, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
leaky relu 3 (LeakyReLU)         (None, 7, 17, 48)     0           dropout_2[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 15, 64)     27712       leaky relu 3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 5, 15, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
leaky relu 4 (LeakyReLU)         (None, 5, 15, 64)     0           dropout_3[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 13, 64)     36928       leaky relu 4[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 3, 13, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 1, 6, 64)      0           dropout_4[0][0]
____________________________________________________________________________________________________
leaky relu 5 (LeakyReLU)         (None, 1, 6, 64)      0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 384)           0           leaky relu 5[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 384)           0           flatten_1[0][0]
____________________________________________________________________________________________________
leaky relu 6 (LeakyReLU)         (None, 384)           0           dropout_5[0][0]
____________________________________________________________________________________________________
hidden1 (Dense)                  (None, 512)           197120      leaky relu 6[0][0]
____________________________________________________________________________________________________
leaky relu 7 (LeakyReLU)         (None, 512)           0           hidden1[0][0]
____________________________________________________________________________________________________
hidden2 (Dense)                  (None, 256)           131328      leaky relu 7[0][0]
____________________________________________________________________________________________________
leaky relu 8 (LeakyReLU)         (None, 256)           0           hidden2[0][0]
____________________________________________________________________________________________________
hidden3 (Dense)                  (None, 50)            12850       leaky relu 8[0][0]
____________________________________________________________________________________________________
leaky relu 9 (LeakyReLU)         (None, 50)            0           hidden3[0][0]
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             51          leaky relu 9[0][0]
====================================================================================================
Total params: 472697
____________________________________________________________________________________________________
```


###### Model more details
So initially I tried many different ways to tune the model.
* I started with using RELU's and then tried ELU's. The one that worked was LeakyRELU.
* As above I also tried to vary the learning rate to tune the model. The one that worked was .0001
* I started with a smaller steering wheel adjustment for left and right images and eventually decided to use .75 factor
* Variations on Dropouts. I tried .4, .2, .3. The one that worked was a combination of .5 and .2.
* Used Adam optimizer
* MaxPooling was used only once, between the last conv layer and the flattened layer with a pool_size of (2, 2)
* Used a biasedCoin method to remove 80% of images with smaller absolute angles:
`def biasedCoin(p):
    return True if random.random() < p else False`
* Initially I chose a value of 7 for epoch. You can see this in the _previous_steps.txt_ file. Here is an example run that did not work (previous_steps.txt):
```
carnd@ip-172-31-43-94:~/behavioral-kera$ python3 p3.py
Using TensorFlow backend.
Learning rate - 0.0001.
Number of epochs - 7.
Adjust left and right angle by a factor of - 0.75
Found 8036 line.
Number of features read from file -  24108
Num of randomly removed 0.0 angle -  6865
Total number of features after -  17243
Number of features after randomly flipping non zero angles - 25929
X_train has 17969 elements. removed 1477
X_valid has 3042 elements. removed 223
X_test has 3019 elements. remove 199
Using generator
starting model
Epoch 1/7
18000/18000 [==============================] - 54s - loss: 0.0224 - val_loss: 0.0176
Epoch 2/7
18000/18000 [==============================] - 53s - loss: 0.0164 - val_loss: 0.0179
Epoch 3/7
18000/18000 [==============================] - 54s - loss: 0.0156 - val_loss: 0.0158
Epoch 4/7
18000/18000 [==============================] - 54s - loss: 0.0154 - val_loss: 0.0168
Epoch 5/7
18000/18000 [==============================] - 53s - loss: 0.0149 - val_loss: 0.0156
Epoch 6/7
18000/18000 [==============================] - 53s - loss: 0.0146 - val_loss: 0.0170
Epoch 7/7
18000/18000 [==============================] - 53s - loss: 0.0145 - val_loss: 0.0163
Test score 0.016528268044273698
Saved model to disk
```
  I then chose for the winning Model to run it as 1 epoch but repeat the same model with shuffled up and changed up data before each run. Below is an example run when things did work. This changing data before each run ensured that things were random and I avoided *overfitting* the model to the data
```
Learning rate - 0.0001.
Number of epochs - 1.
Adjust left and right angle by a factor of - 0.75
Found 8036 line.
Number of features read from file -  24108
Num of extreme angles removed -  10878
Total number of features after -  13230
Number of features after flipping images randomly - 19887
X_train has 14915 elements. removed 0
X_valid has 2485 elements. removed 0
X_test has 2487 elements. remove 0
Using generator
starting model
(Repeat this 10 times) Epoch 1/1
14944/14944 [==============================] - 59s - loss: 0.0262 - val_loss: 0.0232
Test score 0.023195283254608512
Found 8036 line.
Number of features read from file -  24108
Num of extreme angles removed -  10965
Total number of features after -  13143
Number of features after flipping images randomly - 19644
X_train has 14733 elements. removed 0
X_valid has 2455 elements. removed 0
X_test has 2456 elements. remove 0

```

The best result I got was from running 10 cycles, I tried variations, but the 10 was the number (working_model_step.txt)


