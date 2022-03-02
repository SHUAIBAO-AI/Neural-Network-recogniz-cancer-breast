# Neural-Network-recogniz-cancer-breast
Multilayer Perceptron, the basic neural network, can predict the results by supervisor learning. The database is breast-cancer-wisconsin.data,you can download it in Kaggle or by python online.
Lab 5: Neural Network
Shuai BAO
Estimation of Classification Methods
Read the dataset into a list and shuffle it with the random.shuffle method. Hint: fix the random seed (e.g. random.seed(17) ) before calling random.shuffle
Answer:
![image](https://github.com/STPChenFang/Neural-Network-recogniz-cancer-breast/blob/main/IMG-breast%20cancer%20prediction/image005.png)
Figure1:Original dataset(df) and randome shuffled dataset(list)
( 5 marks ) Split the dataset as five parts to do cross-fold validation: Each of 5 subsets was used as test set and the remaining data was used for training. Five subsets were used for testing rotationally to evaluate the classification accuracy. 
Answer:
	According to cross-fold validation theory, we get number of trainset is 556, testset is 139.
 
Figure2:Train set and test set
5.3 	MLP Algorithm
( 5 marks ) All input feature vectors are augmented with the 1 as follows
 
since
 
Answer:
 
Figure3:Augmented dataset
( 5 marks ) Scale linearly the attribute values xij of the data matrix Xˆ into [−1,1] for each dimensional feature as follows:
 
where a small constant 10−6 is used to avoid that the number is divided by zero.
Answer:
 
Figure4:Linear scaled dataset
( 10 marks ) The label ln of the n-th example is converted into a K dimensional vector tn as follows (K is the number of the classes)
 .
Answer:
 
Figure5:Reset label vector
( 10 marks ) Initialize all weight wij of MLP network such as  where
D and K is the number of the input nodes and the output nodes (each node is related to a class), respectively.
Answer:
 
Figure6:Random number-weight ω_ij=ω_input 




( 20 marks ) Choose randomly an input vector x to network and forward propagate through the
network (H is the number of the hidden units)
	
to obtain the error rate  of the example x. Notice that the subscript n in the equations is omitted for the convinence. •
Answer:
 
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
 
Figure8:Initial input layer weight matrix
( 10 marks ) Evaluate the δk for all output units δk = yk – tk
Answer:
According to the previously calculation and the structure of network, we can get the error δk = yk – tk, , δ_k=0.645.
δ_k is the error between train label y_k and predicted result t_k.
The structure of net work is that:
 
Figure9:The sturcture of network
	

( 10 marks ) Backpropagate the δ’s to obtain δj for each hidden unit in the network
 
Answer:
According to the previously calculation, we can get the hidden layer error for each layer:
For hidden layer h1:
 
Figure10:δ_1  in hidden layer h1
For hidden layer h2:
 
Figure11:δ_2  in hidden layer h2
( 10 marks ) The derivative with respect to the first-layer and the second-layer weights are given by
 
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
 
Figure12:Trained input weight matrix ω for input layer
 
Figure13:Trained weight matrix ω bettwen hidden layer h1 and h2
 
Figrue14:Trained weight matrix ω for output layer
( 10 marks ) In the test stage, the test example x is forwarded into the network to obtain the output yK×1 and then assigned to the label with the maximum output value. 
Answer:
	After we got the trained weight matrix network, 



 
Theory and question

 
Flow chart
 
References

 
Code
	# -*- coding: utf-8 -*-  
	""" 
	Created on Sun May 17 20:42:41 2020 
	 
	@author: SHUAI BAO 
	"""  
	import pandas as pd  
	import random #Import shuffle function  
	import numpy as np  
	from collections import  Counter  
	##Define subfunction part  
	  
	#Sample block  
	def split_sample(divide_num,sample_mat):  
	    sample_shape=np.shape(sample_mat) #Take out the matrix size in sample  
	    sample_num=sample_shape[0] #Take out the number of samples in the sample, and assign the value to sample  
	    divide_range=int(sample_num/divide_num) #Divide sample in verage, round down  
	    a=int(divide_range*(divide_num-1)) #Take out the lower bound of sample  
	    b=a+ divide_range#Take the upper bound of sample  
	    train_set=sample_mat[0:a]  
	    test_set=sample_mat[a:b]  
	    return train_set,test_set  
	#End of sample block  
	     
	#Augment vector part:  
	def Augment_feature(augmenting_data):  
	    (sample_num,sample_dim)=np.shape(augmenting_data)  
	    ones=np.ones(sample_num) #Construct row vector 1  
	    ones_column =ones.reshape(-1, 1) #Convert row vector to column vector  
	    augmented_data=np.column_stack((ones_column,augmenting_data[:,0:9],augmenting_data[:,9])) #Add a column 1 to the right of the matrix  
	    return augmented_data  
	#End of enhancement vector part  
	      
	      
	#Linear scaling part, the function name is scale ﹣ linear  
	#Because the data obtained here is in str format,   
	#the exception value will be output at the beginning of index. When calculating formula, STR format needs to be converted to number format  
	def  Scale_linearly(subset):  
	    dimensional_vector=subset[:,0:10]   
	    #Here, there is a class column by default when inputting subset. The class column is in the 11th column, and the 0 column is the enhancement vector 1  
	    sample_shape=np.shape(dimensional_vector) #Take out the matrix size in sample  
	    sample_dimension=sample_shape[1] #Take out the number of dimensions in the sample  
	    for j in range(sample_dimension):   
	        vector=dimensional_vector[:,j]  
	        #vector=[int(i) for i in vector_str] #Convert STR data to number type  
	        min_j=min(vector)#Find min value in vector  
	        min_j=float(min_j)  
	        max_j=max(vector)#Find max value in vector  
	        max_j=float(max_j)  
	        vector=np.mat(vector)  
	        dimensional_vector[:,j]=2*(vector-min_j+10e-6)/(max_j-min_j+10e-6)-1  
	    Xij=np.hstack((dimensional_vector,subset[:,10:11]))   
	    #The returned data contains the combination of 10 dimensions after linear scaling and the last class column, a total of 11 columns  
	    return Xij  
	#End of linear scaling section  
	      
	#Reset sample vector x according to label y  
	def Reset_example_vector(subset):  
	    subset_num_mat=np.shape(subset)  
	    subset_num=subset_num_mat[0]  
	    Xij=subset  
	    for k in range(subset_num):  
	        if subset[k,10]==4: #k=ln defined as 1  
	            Xij[k,10]=1  
	        else:  
	            Xij[k,10]=0 #k!=ln defined as 0  
	    x=Xij  
	    return x  
	#End of Reset sample vector x according to label y  
	      
	#tanh（）function   
	def tanh(x):  
	    Xmat = np.mat(x, dtype=float) #Define the parameter after mat data as float, otherwise the accounting calculation will report an error  
	    s1 = np.exp(Xmat) - np.exp(-Xmat)  
	    s2 = np.exp(Xmat) + np.exp(-Xmat)  
	    s = s1 / s2  
	    return s  
	#End of tanh()  
	      
	#Define hidden layer function, weight_ W_ Input is the weight matrix of this layer, input_ Xi is the underlying node value vector, hidden_ node_ Vector is the node value vector of this layer,  
	def hidden_layer(input_xi,weight_w_input):   
	    #The number of neurons in the input layer and the next hidden layer is h = 9  
	    (hidden_node_num,input_dimension)=np.shape(weight_w_input) #Take out the number of nodes and input vector dimension of weight matrix  
	    hidden_node_vector=np.zeros(hidden_node_num) #Initialize node value vector of this layer  
	    input_xi=np.array(input_xi) #Convert data format to array for subsequent multiplication   
	    for node_num_i in range(hidden_node_num):#Calculate the vector of node sequence in this layer  
	        w_i=weight_w_input[node_num_i] #Remove the node_ num_ I weight vectors to calculate the next level node_ num_ Values of I nodes  
	        input_weight_mat=input_xi*w_i #Calculate the element product of input layer and weight vector to get input_ weight_ Mat  
	        input_weight_num=np.sum(input_weight_mat) #Calculate the product sum of input layers, input_ weight_ Num is the sum of the product of input layer and weight vector  
	        input_weight_num_activate=tanh(input_weight_num) #The input layer is activated, and the activation function is tanh ()  
	        hidden_node_vector[node_num_i]=input_weight_num_activate #Calculate the node value and assign it to the point vector of this layer  
	    #hidden_node_sum=np.sum(hidden_node_vector) #Sum of vector values of all nodes  
	    return hidden_node_vector #Output node value vector of this layer  
	#End of hidden layer function definition  
	  
	#定BP function, weight_ W_ now_ Layer is the weight vector of the current layer,   
	#Delta_ K is the total calculation error, and the default calculation function is tanh(),   
	#now_ layer_ node_ Num is the sum of the underlying calculation, that is, the value of the current node  
	def BP_error(weight_w_now_layer,delta_k,now_layer_node_num):#Define BP back propagation calculation error value  
	    weight_error_mat=weight_w_now_layer*delta_k  
	    weight_error_sum=np.sum(weight_error_mat,axis=1) #Sum each row, and then the column vector  
	    deuction_delta=1-now_layer_node_num**2  
	    BP_delta_j=(deuction_delta)*weight_error_sum  
	    return BP_delta_j  
	#End of BP function definition  
	  
	#Define update weight vector function  
	#Xi is the bottom layer node value vector,   
	#Delta_ layer_ 1 is the error vector of each node at the top level, a sequence,   
	#ETA_ learning_ Rate is the super parameter of learning rate and custom setting  
	#previous_weight_vector是未更新前的权重向量矩阵  
	def update_weight_vector(xi,delta_layer_1,eta_learning_rate,previous_weight_vector):  
	    delta_layer_1=np.mat(delta_layer_1)  
	    [layer1_col,layer1_node_num]=np.shape(delta_layer_1)  
	    xi_dimension=np.size(xi)  
	    new_weight_vector=np.zeros((layer1_node_num,xi_dimension))  
	    for layer1_node_num_i in range(layer1_node_num): #Update the underlying weight vector corresponding to this node based on the deviation value of each node  
	        layer1_delta_i=delta_layer_1[0,layer1_node_num_i] #Take out layer1 layer layer1_ node_ num_ Error of I nodes  
	        new_weight_vector[layer1_node_num_i]=previous_weight_vector[layer1_node_num_i]-eta_learning_rate*layer1_delta_i*xi  
	    return  new_weight_vector  
	#Define update weight vector function end  
	      
	def forward_calculate(input_xi_label,weight_w_input,weight_w_1,weight_w_output):  
	    #前向计算开始  
	    #weight_w_input=weight_wij_input_temp #Input layer weight vector  
	    #weight_w_1=weight_wij_2_temp #Hidden layer weight vector output from the first layer to the second layer  
	    #weight_w_output=weight_wij_output_temp #Hidden layer to output layer weight vector  
	    input_xi=input_xi_label[0:10] #Take 0 to 10 numbers in the sample as the data vector, the 11th number as the result judgment, and the X matrix as the input  
	    #Forward calculation start  
	    #The number of neurons in the input layer and the next hidden layer is h = 9  
	    (hidden_node_num,input_dimension)=np.shape(weight_w_input) #Take out the number of nodes and input vector dimension of weight matrix  
	      
	    input_xi=np.array(input_xi) #Convert data format to array for subsequent multiplication   
	      
	    for node_num_i in range(hidden_node_num):#Calculate the vector of the next level node sequence  
	        w_i=weight_w_input[node_num_i] #Remove the node_ num_ I weight vectors to calculate the next level node_ num_ Values of I nodes  
	        input_weight_mat=input_xi*w_i #Calculate the element product of input layer and weight vector to get input_ weight_ Mat  
	        input_weight_num=np.sum(input_weight_mat) #Calculate the product sum of input layers, input_ weight_ Num is the sum of the product of input layer and weight vector  
	        input_weight_num_activate=tanh(input_weight_num) #The input layer is activated, and the activation function is tanh ()  
	        hidden_node_vector[node_num_i]=input_weight_num_activate #Calculate the node value and assign it to the next node vector  
	    hidden_layer_1_node_num=hidden_node_vector #Update the node value of the next layer, and the activation function is tanh ()  
	    #End of input layer  
	      
	    #First hidden layer  
	    #Input calculation_ weight_ num_ Activate as the first input vector of hidden layer, hidden_ layer_ Num1 is the output vector of the first hidden layer  
	    hidden_layer_num1_activate=hidden_layer(hidden_layer_1_node_num,weight_w_1)  
	    #hidden_layer_num1_activate=tanh(hidden_layer_num1)   
	    #Input calculation_ weight_ num_ Activate as the first input vector of hidden layer, hidden_ layer_ Num1 is the output vector of the first hidden layer  
	      
	    #The second hidden layer  
	    #Input calculation, hidden_ layer_ num1_ Activate as the second input vector of hidden layer, hidden_ layer_ 2_ node_ Num is the output vector of the second hidden layer  
	    hidden_layer_2_node_num=hidden_layer_num1_activate  
	    hidden_layer_num2_activate=hidden_layer(hidden_layer_2_node_num,weight_w_output)  
	    #hidden_layer_num2_activate=tanh(hidden_layer_num2)  
	    #Hidden layer end of layer 2  
	    #Output layer  
	    reult_yk=hidden_layer_num2_activate #Get output result_ YK  
	    #End of forward calculation  
	    delta_k=reult_yk-input_xi_label[10]  
	    return reult_yk,delta_k  
	    #End of forward calculation  
	  
	  
	      
	#Import data  
	df = pd.read_csv(r'C:\Users\SHUAI BAO\OneDrive\2020年\西交利物浦大学\EEE418高级特征识别Advanced Pattern Recognition\Lab\lab5-neural-network\breast-cancer-wisconsin.data') # 读取数据集  
	#When using the import statement, add r before the path to indicate that the string is a non escaped original string  
	#End of importing  
	      
	#problem5.2.1：Random shuffle data  
	list_mat=np.mat(df) #Convert the data in DF parameter from panda.core.frame.dataframe to matrix type, and then name it list  
	list=np.array(list_mat) #Mat format to array format  
	list[list == '?'] = 0 #For data cleaning, the data inside is'? ' Replace the data of with 0  
	(row_num,col_num)=np.shape(list)  
	for mat_num in range(row_num):  
	    list[mat_num,6]=float(list[mat_num,6]) #Change STR data in data to num format  
	list=list[:,1:11]  
	random.seed(17) #Define the integer value at the beginning of the algorithm before calling the random module  
	#导入后的数据命名为list  
	np.random.shuffle(list) #Arrange the list data randomly. Random is only valid for list type and not for array types  
	##problem1：End of Random shuffle data  
	  
	#problem5.2.2：Split data  
	divide_num=5 #Define the number of splitting  
	(train_set,test_set)=split_sample(divide_num,list)  
	#problem2：End of splitting  
	  
	#problem5.3.1：Augment the data  
	X_bar_temp=Augment_feature(train_set)  
	X_augmented=np.mat(X_bar_temp)  
	#problem5.3.1：End of Augment the data5  
	  
	#problem5.3.2：Linearly scale  
	X_data_scaling=X_bar_temp   
	#The default input data for linear scaling is 10 columns of data plus the 11th column of label, 11 columns in total, and the first column is the enhanced vector column 1  
	Xij_scale=Scale_linearly(X_data_scaling) #The output data after linear scaling is 10 columns of data plus the 11th column of label, 11 columns in total  
	#problem5.3.2：End of Linearly scale  
	  
	#problem5.3.3： Reset the example vector x according its label y  
	Xij_scale_input=np.array(Xij_scale)  
	x=Reset_example_vector(Xij_scale_input)  
	#problem5.3.3： End of Reset the example vector x according its label y  
	  
	  
	#Initialization Wij, D is the number of input nodes, K is the number of output nodes  
	(sample_num,sample_dimension)= np.shape(x) #Take out the number of dimensions and samples of X matrix  
	D_inputnode=sample_dimension #The input node is the number of characteristic dimensions  
	K_outputnode=1 #Number of output nodes is 1  
	random_area=np.sqrt(6/(D_inputnode+K_outputnode+1)) #Random number range calculation  
	#Initialize weight vector Wij matrix  
	H=9 #Number of hidden layer nodes  
	weight_wij_input=-random_area + 2*random_area*np.random.random((H,sample_dimension-1))   
	#Using random number to generate Wij 1, initializing the matrix of 60 * 10, 60 is the number of nodes, 10 is the number of dimensions of input data  
	weight_wij_2=-random_area + 2*random_area*np.random.random((H,H))   
	#Using random number to generate Wij 2, initializing it into matrix, the link between hidden layer 1 and hidden layer 2  
	weight_wij_output=-random_area + 2*random_area*np.random.random((1,H))   
	#Random number generates the weight matrix between hidden layer and output layer, and the weight is vector  
	#Generating Wij 2 with random number and initializing it into sequence  
	weight_wij_input_temp=weight_wij_input #Initialize and then update the staging value of the node  
	weight_wij_2_temp=weight_wij_2  
	weight_wij_output_temp=weight_wij_output  
	hidden_node_vector=np.zeros(H) #Initialize hidden layer node vector  
	#End of initialization  
	  
	#BP update node weight, X is the input value, hidden layer network is 2  
	for sample_i in range(sample_num):  
	    weight_w_input=weight_wij_input_temp #Input layer weight vector  
	    weight_w_1=weight_wij_2_temp #Hidden layer weight vector output from the first layer to the second layer  
	    weight_w_output=weight_wij_output_temp #Hidden layer to output layer weight vector  
	    #Forward calculation start  
	    #The number of neurons in the input layer and the next hidden layer is h = 9  
	    (hidden_node_num,input_dimension)=np.shape(weight_w_input) #Take out the number of nodes and input vector dimension of weight matrix  
	    input_xi=x[sample_i,0:10] #Take 0 to 10 numbers in the sample as the data vector, the 11th number as the result judgment, and the X matrix as the input  
	    input_xi=np.array(input_xi) #Convert data format to array for subsequent multiplication    
	      
	    for node_num_i in range(hidden_node_num):#Calculate the vector of the next level node sequence  
	        w_i=weight_w_input[node_num_i] #Remove the node_ num_ I weight vectors to calculate the next level node_ num_ Values of I nodes  
	        input_weight_mat=input_xi*w_i #Calculate the element product of input layer and weight vector to get input_ weight_ Mat  
	        input_weight_num=np.sum(input_weight_mat) #Calculate the product sum of input layers, input_ weight_ Num is the sum of the product of input layer and weight vector  
	        input_weight_num_activate=tanh(input_weight_num) #The input layer is activated, and the activation function is tanh ()  
	        hidden_node_vector[node_num_i]=input_weight_num_activate #Calculate the node value and assign it to the next node vector  
	    hidden_layer_1_node_num=hidden_node_vector #Update the node value of the next layer, and the activation function is tanh ()  
	    #End of input layer  
	      
	    #First hidden layer  
	    #Input calculation_ weight_ num_ Activate as the first input vector of hidden layer, hidden_ layer_ Num1 is the output vector of the first hidden layer  
	    hidden_layer_num1_activate=hidden_layer(hidden_layer_1_node_num,weight_w_1)  
	    #Hidden layer first layer end  
	      
	    #The second hidden layer  
	    #Input calculation, hidden_ layer_ num1_ Activate as the second input vector of hidden layer, hidden_ layer_ 2_ node_ Num is the output vector of the second hidden layer  
	    hidden_layer_2_node_num=hidden_layer_num1_activate  
	    hidden_layer_num2_activate=hidden_layer(hidden_layer_2_node_num,weight_w_output)  
	    #Hidden layer end of layer 2  
	    #Output layer  
	      
	    reult_yk=hidden_layer_num2_activate #Output result_yk  
	    #End of forward calculation  
	      
	    #Back propagation update section  
	    #Calculate the error value of each layer and each node  
	    tk=x[sample_i,10] #Take this sample input_ The actual judgment value of Xi, located in the last column of the sequence, is named TK,  
	    delta_k=reult_yk-tk #YK is the calculated result, TK is the original result, Delta_ K is the total difference  
	    delta_k=delta_k[0] #delta_ K is a number  
	    #delta_k=np.sqrt(delta_k**2)  
	    delta_layer_2=BP_error(weight_w_output,delta_k,hidden_layer_2_node_num)   
	    #weight_ W_ Output is the weight from this layer to the next layer, hidden_ layer_ num2_ Activate is the output value of the bottom layer, Delta_ K total difference  
	    hidden_layer_1_node_num=np.array(hidden_layer_1_node_num) #Convert to array format data for subsequent calculation  
	    delta_layer_1=BP_error(weight_w_1,delta_k,hidden_layer_1_node_num) #First hidden layer error vector, Delta_ layer_1  
	    #Calculate the number of back propagation closest to the output layer  
	      
	    #Update node weight vector  
	    eta_learning_rate=0.005 #User defined learning rate ETA, ETA is a super parameter  
	    #previous_ weight_ Vector is the weight vector of update money  
	    weight_wij_input_temp=update_weight_vector(input_xi,delta_layer_1,eta_learning_rate,weight_w_input)#Update input weight vector  
	    weight_wij_2_temp=update_weight_vector(hidden_layer_1_node_num,delta_layer_2,eta_learning_rate,weight_w_1)#Update the first layer weight vector of hidden layer  
	    delta_layer_3=np.array(tk)  
	    weight_wij_output_temp=weight_wij_output_temp-eta_learning_rate*delta_layer_3*hidden_layer_num2_activate  
	#BP update weight end  
	      
	#Test set calculation accuracy  
	  
	X_bar_test=Augment_feature(test_set) #Enhanced test matrix  
	X_test_augmented=np.mat(X_bar_test) #Strengthening matrix matrix  
	X_data_scaling=X_test_augmented  
	X_test_scale=Scale_linearly(X_data_scaling) #Linear scaling enhancement vector  
	X_test_scale_input=np.array(X_test_scale)  
	x_test=Reset_example_vector(X_test_scale_input) #Linear zoom end, X_ Test as test data set  
	  
	#BP network prediction  
	(test_num,test_dimension)= np.shape(x_test) #Take out x_ Dimension number and sample number of test matrix  
	result_predict=np.zeros(test_num)  
	#H=9   
	#weight_w_input  
	#weight_w_1  
	#weight_w_output  
	#BP forward prediction calculation, X is the input value, hidden layer network is 9  
	predict_delta=np.zeros(test_num) #Initialize prediction error sequence  
	for test_i in range(test_num):  
	    [result_predict[test_i],predict_delta[test_i]]=forward_calculate(x_test[test_i],weight_w_input,weight_w_1,weight_w_output)  
	result_predict_abs=np.sqrt(result_predict**2) #Take the absolute value of the result sequence  
	#End of forecast calculation  
	for result_i in range(test_num):  
	    if result_predict_abs[result_i]>0.5:  
	        result_predict_abs[result_i]=1  
	    else:  
	        result_predict_abs[result_i]=0  
	result_real=x_test[:,10] #Extract test set result vector  
	matched_vector=result_real-result_predict_abs #Calculate the result sequence with correct prediction, and the element value of 0 means correct prediction  
	  
	#Calculation accuracy part  
	result_set=Counter(matched_vector)  
	correct_num_train=result_set[0.0]  
	correct_rate=correct_num_train/test_num #Calculate forecast accuracy  
	print('The accuracy of test set is',correct_rate,'')  
	#End of calculation accuracy  

