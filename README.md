# Neural-Network-recogniz-cancer-breast
Multilayer Perceptron, the basic neural network, can predict the results by supervisor learning. The database is breast-cancer-wisconsin.data,you can download it in Kaggle or by python online.
Lab 5: Neural Network
Shuai BAO
Estimation of Classification Methods

Read the dataset into a list and shuffle it with the random.shuffle method. Hint: fix the random seed (e.g. random.seed(17) ) before calling random.shuffle

Answer:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image005.png)

Figure1:Original dataset(df) and randome shuffled dataset(list)

Split the dataset as five parts to do cross-fold validation: Each of 5 subsets was used as test set and the remaining data was used for training. Five subsets were used for testing rotationally to evaluate the classification accuracy. 
Answer:
According to cross-fold validation theory, we get number of trainset is 556, testset is 139.

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image009.png)

Figure2:Train set and test set

MLP Algorithm
All input feature vectors are augmented with the 1 as follows  since 
Answer:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image017.png)

Figure3:Augmented dataset

Scale linearly the attribute values xij of the data matrix Xˆ into [−1,1] for each dimensional feature as follows:
where a small constant 10−6 is used to avoid that the number is divided by zero.
Answer:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image023.png)

Figure4:Linear scaled dataset

The label ln of the n-th example is converted into a K dimensional vector tn as follows (K is the number of the classes)
Answer:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image029.png)

Figure5:Reset label vector

Initialize all weight wij of MLP network such as  where
D and K is the number of the input nodes and the output nodes (each node is related to a class), respectively.
Answer:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image169.png)

Figure6:Random number-weight ω_ij=ω_input 


Choose randomly an input vector x to network and forward propagate through the network (H is the number of the hidden units) to obtain the error rate  of the example x. Notice that the subscript n in the equations is omitted for the convinence. •
Answer:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image012.png)

Figure7:Structure of full connect between hidden layer and inpute layer
	The input vector is X_i=[x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10 ]  , for x_1, the next hidden layer unit matrix is 
Weight_matrix=[■(ω_11&⋯&ω_(1 10)@⋮&⋱&⋮@ω_(10 1)&⋯&ω_(10 10) )]
	So for hidden layer h1-unit1-unactivate, we can get the value of it: 
Hidden layer h1-unit1-unactivate=f(x_i )=∑x_i×ω_1i=x_1×ω_11+x_2×ω_12+x_3×ω_13+x_4×ω_14+x_5×ω_15+x_6×ω_16+x_7×ω_17+x_8×ω_18+x_9×ω_19+x_10×ω_(1 10)
Hidden layer h1-unit2-unactivate=f(x_i )=∑x_i×ω_2i=x_1×ω_21+x_2×ω_22+x_3×ω_23+x_4×ω_24+x_5×ω_25+x_6×ω_26+x_7×ω_27+x_8×ω_28+x_9×ω_29+x_10×ω_(2 10)
…
Hidden layer h1-unit9-unactivate=f(x_i )=∑x_i×ω_9i=x_1×ω_91+x_2×ω_92+x_3×ω_93+x_4×ω_94+x_5×ω_95+x_6×ω_96+x_7×ω_97+x_8×ω_98+x_9×ω_99+x_10×ω_(9 10)
Hidden layer h1-unit10-unactivate=f(x_i )=∑x_i×ω_10i=x_1×ω_(10 1)+x_2×ω_(10 2)+x_3×ω_(10 3)+x_4×ω_(10 4)+x_5×ω_(10 5)+x_6×ω_(10 6)+x_7×ω_(10 7)+x_8×ω_(10 8)+x_9×ω_(10 9)+x_10×ω_(10 10)
In summary, we can get the matrix calculation of weight from input x to hidden layer h1:
Hidden layer h1-unit-unactivate=X_i×Weight_matrix^'=[x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10]×[■(ω_11&⋯&ω_(10 1)@⋮&⋱&⋮@ω_(1 10)&⋯&ω_(10 10) )]
After we got the hidden layer vector, we should use activate function to get new unit value in hidden layer h1:
Activate function:tanh⁡(x)=(e^x-e^(-x))/(e^x+e^(-x) )  
Hidden layer h1-unit=tanh⁡(Hidden layer h1-unit-unactivate)
	In this sample, we use random value to initialize the weight matrix:
	
![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image169.png)

Figure8:Initial input layer weight matrix

Evaluate the δk for all output units δk = yk – tk
Answer:
According to the previously calculation and the structure of network, we can get the error δk = yk – tk, , δ_k=0.645.
δ_k is the error between train label y_k and predicted result t_k.
The structure of net work is that:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image134.png)

Figure9:The sturcture of network

Backpropagate the δ’s to obtain δj for each hidden unit in the network
Answer:
According to the previously calculation, we can get the hidden layer error for each layer:
For hidden layer h1:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image147.png)

Figure10:δ_1  in hidden layer h1
For hidden layer h2:

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image151.png)

Figure11:δ_2  in hidden layer h2
The derivative with respect to the first-layer and the second-layer weights are given by
 
The framework of MLP algorithm is as follows, where η = 0.001. Note that η, T and H are the hyperparameters of the network.
Algorithm 1 Stochastic Backpropagation Algorithm	
1: Initialize w, η
2: for t = 1 to T do
3:	Shuffle the training data set randomly.	
4:	for n = 1 to N do	
5:	Choose the input xn	
6:	Forward the input xn through the network	
7:	Backward the gradient from the output layer through network to obtain  	and 
8:	Update the weights of the network
 	
9:
10:
11:	end for
end for return w	
The algorithm may be terminated by setting the total iteration T except that setting the threshold θ of the gradient referred in the lecture slide.
Answer:
Repeat the network to train the weight on trainset. And after we get the final weight matrix, the total iteration T is from sample number of trainset,T=556, threshold θ=0.5.
We get the result :
![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image169.png)
Figure12:Trained input weight matrix ω for input layer
![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image173.png)
Figure13:Trained weight matrix ω bettwen hidden layer h1 and h2
![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image151.png)
Figrue14:Trained weight matrix ω for output layer
In the test stage, the test example x is forwarded into the network to obtain the output yK×1 and then assigned to the label with the maximum output value. 
Answer:
After we got the trained weight matrix network, 



 
Theory and question
About 𝜼: learning rate
Learning rate(𝜂) is also called step size, which is set as constant in standard BP algorithm.
However, in practice, it is difficult to use a certain value as the best learning rate. If the learning rate
can be adjusted dynamically, it is good. From the weight surface, we hope it will increase in the flat
area, because too small will increase the number of training, and increase will accelerate the
separation from the flat area. In the area with large error variation, it is easy to cross a narrow lowest
point. This lowest point may be the best of the whole, and it will generate oscillation, but the number
of iterations will increase. Therefore, in order to accelerate convergence, a better solution is to
dynamically adjust the learning rate according to the specific situation. Here is an implementation
method:
The study of adaptive learning rate is a field, which is initially explored from the annealing
algorithm to the optimal annealing algorithm. The optimal annealing algorithm provides online
learning, which is an important step for online learning. However, the disadvantage of the annealing
scheme is that the time constant is a priori. Considering the practical problems, different sample priori
will change, so Murata first proposed it in 1998 The online learning algorithm needs to be equipped
with the internal mechanism for the adaptive control of learning rate, and the learning of the learning
algorithm has been properly modified. The first correction is when the statistical characteristics
change, and the second is to increase the generalization ability of the online learning algorithm. So
the prior problem of annealing algorithm is solved. However, it is at the cost of suboptimal solution in
the annealing range considering the learning rate parameter. Its important advantage is that it
expands the applicability of online learning in the actual implementation mode.
Set an initial learning rate. If the total error E increases after a batch of weight adjustment, it
means that this adjustment has no effect. The reason is that the learning rate may be too large and
oscillate, so the learning rate needs to be reduced: 𝜂(𝑇 + 1) = 𝛽 𝜂(𝑡) (𝛽 < 1) If the total error E
is reduced after a batch of weight adjustment, the adjustment is effective. At the same time, the
iteration speed can be accelerated, that is, the learning rate can be increased: 𝜂(𝑇 + 1) =
𝜃 𝜂(𝑇) (𝜃 > 1),
About hidden layer and nodes
In general, we can decrease the network error by adding the number of hidden layer, but in
our lab, the number of layer was given to 2. The more hidden layer is , the more accurate the result
is , but when we start to train the network, we may overfit it and take more time to train it.
So, on the one hand , we can get the lower error by more hidden layer units, it’s easier to
accomplish than increase the number of hidden layers.
In BP net, it’s significant to determine the number of hidden units in hidden layer, until now,
there is no theory to help us determine the number.
Most of the formulas for determining the number of hidden layer nodes proposed in the
previous literature are for any number of training samples, and most of them are for the most
unfavorable situation, which is difficult to meet in general engineering practice and should not be
used. In fact, the number of hidden layer nodes obtained by various calculation formulas
sometimes varies several times or even hundreds of times.
In order to avoid the phenomenon of "over fitting" in training as much as possible, and ensure
enough high network performance and generalization ability, the most basic principle for
determining the number of hidden layer nodes is to take the compact structure as much as possible
on the premise of meeting the accuracy requirements, that is, to take the minimum number of
hidden layer nodes.
The results show that the number of hidden layer nodes is not only related to the number of
input / output layer nodes, but also related to the complexity of the problems to be solved, the type
of conversion function and the characteristics of sample data.
We must meet the condition below before we determine the number of nodes: es.
1. Number of hidden layer nodes must less than the number of train sample. Otherwise, the
system error of the network model is independent of the characteristics of the training samples
and tends to zero, that is to say, the established network model has no generalization ability
and no practical value. Similarly, it can be deduced that the number of nodes (variables) in the
input layer must be less than n-1
2. The number of training samples must be more than the connection weight of the network
model, generally 2-10 times. Otherwise, the samples must be divided into several parts and
"training in turn" method can be used to get a reliable neural network model.
In summary, if the number of hidden layer nodes is too small, the network may not be able to train
at all or the network performance is poor; if the number of hidden layer nodes is too large,
although it can reduce the system error of the network, on the one hand, it can prolong the training
time of the network, on the other hand, the training is easy to fall into the local minimum and can
not get the best, which is also the internal reason of "over fitting" in training. Therefore, the
reasonable number of hidden layer nodes should be determined by node deletion method and
expansion method, considering the complexity and error of network structure

Flow chart

![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image174.png)

