# Deep_into_CNNs


## Week 1

### ● Plan/Goals

```
○ Numerical data : Multi layer Perceptron ( MLP ) :
■ Regression python Implementation
■ Gradient Descent,relu layer, MSE loss
■ Binary Classification,sigmoid layer,BCE loss
■ Multiclass Classification,softmax layer
```
```
○ NLL loss MLP + PyTorch :
■ Linear Algebra, Single Layer NN, Training, - Inferenceand Validation :
Illustrated Through Pytorch
■ Implement 1-hidden layer NN using PyTorch but trainin python
```
### ● Tasks

```
○ Content reading on Regression And Shallow NN UsingPython.
```
```
■ Things Learnt:
```
```
● Cost/Loss function: Structure and the basic use ofcost/loss
functions were taught
```
```
● (Stochastic) Gradient descent: It includes the processof linear
descent and how it works. It also included the limitand use of
parameters like learnrate while implementing gradientdescent
```
```
● Python Implementation from scratch : All the abovethings
were taught in python without using any big library.It was done to
give a clear picture of how different things/functionswork and
implemented directly.
```

```
○ Completing the programming exercises shared and updating github repo
with practice code and completed exercises.
```
```
■ Things learnt:
```
```
● Basic data handling with numpy and pandas : Menteeswere
taught how to load data from csv and clean the datawith the help
of numpy and pandas. Other than the basic functions,mentees
were also taught One-hot encoding of the data, normalisations
and its importance,and some implementations of matplotlibto
view data.
```
```
● Implementing sigmoid and error calculation functions :
sigmoid and error calculation functions were taughtand
implemented with numpy.
```
```
● Training and implementing shallow NN from scratchin
python: After all the functions were made, mentees
implemented the them along with error term calculationand back-
propagation to make a shallow NN and there accuracyand loss
were observed after every epochs
```
## Week 2

### ● Plans/Goals

```
○ Intro to CNN :
■ Simple Feed-forward Network :
● Flatten images first and then treat them as numericaldata.
```
```
■ Convolutional Neural Networks :
● Use Spatial Information
```
```
■ Compare results with MLP on MNIST data
■ Start Using PyTorch.
```
### ● Tasks

```
○ Content reading on Neural Networks in pytorch
```
```
■ Things learnt -
```

```
● Backprop: Though it was already implemented in week1, but it
was used only in shallow NN and therefore it was includedagain
in week 2 to give a complete sense of its implementationon deep
NN. Also, this time it was implemented using pytorch.
```
```
● Softmax : Mentees were taught about the use, importanceand
implementation of softmax functions from scratch andin pytorch.
```
```
● Basics of Pytorch : Assignments were given to teachmentees the
implementation of different functions (related toNN) in pytorch.
```
```
○ Complete the programming exercises shared and updategithub repo with
practice code and solved assignments: Total of 8 assignmentswere given.
Mentees were taught how to -
```
```
● Load and handle data using pytorch
● Uses and implementation of dataloaders and importanceof
parameters like batch-size
● Loading already available data using torch visionand the process
of normalisation
● Basic structure and documentation about the NN inpytorch
● Criterions like CrossEntropyLoss, NLLLoss,
● Optimizer like Adam, SGD
● Use, importance and complete implementation of pytorch
autograd and its use with loss functions.
● Train neural networks and setting of hyperparametersand
learning rate to adjust data along with RELu and LogSoftmax
functions
● Importance of Validating the data during trainingand its
implementation using the Dropout function.
● Saving and loading back the already trained modelsand its
importance
● An optional exercise to make a Cat-Dog identifierwas also given
along with the basic framework required to make it.
```
```
○ Hackathon-1 starts : The Hackathon 1 data set was verynoisy and was given to
give an idea of how Kaggle and Hackathon works. Italso signified an important
fact that simple fully connected NN can sometimesbe inefficient to train and
predict the data accurately.
```
## Week 3


### ● Plans/Goals

### ○ Layers: Maxpool, Average and Dropout ,Fully connectedlayers in

### combination with Convolutional layers.

```
○ LeNet:
■ Convolution
■ Pooling
■ Fully connected layers
```
```
○ Competition of Hackathon 1
```
```
○ Practice assignments on CNN
```
### ● Tasks

```
○ Hackathon-1 submission : The hackathon was an openone. Hosted on kaggle
and included a total of over 800 teams (Tabular PlaygroundSeries - Jun 2021).
Following were the top 5 scores of mentees along withtheir scores:
```
```
○ Hackathon-2 starts: It is based on training NN andmaking predictions for RGB
images.
```
```
○ Read a famous shared paper and update the github repowith practice code
and solved implementation of that paper. Mentees wereasked to choose one
of the following SOTA models on ImageNET classificationpapers and implement
it.
```
### Paper Implementaion

1. AlexNet:
https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
2. VGG:
https://arxiv.org/pdf/1409.1556v6.pdf
3. Inception(GoogLeNet):
https://arxiv.org/pdf/1409.4842v1.pdf
4. Xception:
https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf
5. ResNet:
https://arxiv.org/pdf/1512.03385v1.pdf

```
● Filters : Creating filters(edge detection, sobel,customised)and applying them
to images.
● Visualize Convolution : Visualize four filtered outputs(a.k.a. activation
maps) of a convolutional layer.
● MNIST : Train an MLP to classify images from theMNISTdatabase
hand-written digit database.
```

```
● Different types of layers : Train a CNN to classify images from the CIFAR-
databaseand definea CNN architecture using Convolutionallayers and Max
Pooling layers.
● Hyperparameter tuning : Deciding a loss and optimizationfunction that is
best suited for this classification task.
● LeNet-5 : Implement a modified LeNet-5 for MNIST digitsclassification but
use max instead of average pooling and use fully connectedinstead of
gaussian final layer.
```
## Week 4 - Week

### ● Plans/Goals

```
○ Optimizer variation:
■ SGD with Momentum, Nesterov and Adam
```
```
○ Overfitting and Regularization
■ L1, L
■ Batch-Norm
```
```
○ Hyperparameter tuning
■ Variable learning rate
■ Weight Initialization : Xavier, He Normal
```
### ● Tasks

### ○ Paper 1 Submission: Out of the four given papers onSOTA models ( VGG,

```
Inception, Exception. AlexNet), mentees were askedto choose one and
implement it.
```
```
■ Things Learnt:
● The thinking process: Mentees learnt about the processof
“deciding” of where and how the layers should be addedto
improve the efficiency/accuracy of the model.
```
```
● New Ideas: The different and innovative ways to implementan
idea in a model, for example - the splitting of layersof data and
making them pass through different processing layersand
combining them again in the end, was never taughtto mentees
before.
```

```
● The practical use of Optimizers and Batch-normalization:
Mentees were able to see the use of optimizers, non-linearity
along with batch-normalization (that was taught thisweek) in
action through these papers
```
```
○ Hackathon-2 ends: Hackathon was based on the followingdata-set-
```
### ■ https://www.kaggle.com/gpiosenka/100-bird-species

```
It was a data set consisting of RGB images of birdsdivided into 275 species.
The Mentees were advised to use their SOTA modelson this data set along with
the optimisation techniques taught in this week.
```
```
■ Things Learnt:
```
```
● Data modification: Though the data was already clean,mentees
were still supposed to apply transformation to itto make it more
general- a step to prevent overfitting.
```
```
● Practice on implementing the SOTA models on a dataset that
requires a deep model and is difficult to get goodaccuracy on.
```
```
● Self-Implementation of optimization techniques: Thementees
were able to use the techniques to find the best parametersto fit
the data-set. They were also encouraged to slightlymodify the
SOTA model to introduce dropouts and other thingsthat they
thought fit.
```
## Week 5 - Week 6

### ● Plans/Goals:

```
○ Autoencoders
```
```
■ Convert High dimension to Low dimension data
```

```
■ Should be able to convert Low to high with minimum error
```
```
■ MLP
● First flatten images i.e. convert to numerical data
● As a (ineffective) compression method
```
```
■ Convolution
● For Denoising images
● Uses Transposed Convolutions
```
```
○ Generative Adversarial networks
```
```
■ Generate new data points as efficiently possible Generator
■ Generate fake data Discriminator
■ Recognize fake data and penalize generator
■ Generator and Discriminators Compete with Each Other!!!
```
### ● Tasks: Due to the time constraints ( partially dueto the break for Y20 End- Sem and

```
partially due to the Y19 Internships), the week 5- week 6 plans were cut short and only
the following task was given -
```
### ○ Complete the programming exercises shared and updatethe github repo

```
with practice code and solved assignments- therewere a total of 3
assignments. Mentees were taught the following things-
```
```
■ Simple encoding and decoding (reproducing) of datausing a neural
network.
```
```
■ Repeating the above process by adding convolutioninstead of simple
neural network and observing improvement in results
```
```
■ Adding a lot of noise in the input data and makingthe model deeper with
the goal of de-noising the data, by setting the cleandata as the target of
noisy data for the model.
Example of the output - the above layer is the inputand the second
layer is the output-
```

