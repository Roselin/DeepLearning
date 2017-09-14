# **HW3 - Yun-Jou Lin 0027875836** 

##Results
I have done two tests (1) Check back propogation and (2) Logical Gate for every training
https://github.com/Roselin/DeepLearning/blob/master/HW3.PNG

###(1) Check back propogation

I used the example provided on the class to test the back propagation  
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/  
You can get the exactly same value  

###(2) Test Logical Gate

I the original theta value I designed from last course is  
And -> (-50, 40, 40)'
Or ->  (-20,30,30)'
Not -> (10, -20)'
XOR -> first layer ([-50, 80, -70],[-50,-70,80])'
       second layer (-50,40,40)

Compared with this time 
And -> (-11.261, 7.3558, 7.2609)
Or -> (-3.8304, 8.0144, 8.1142)
Not -> (4.4764, -9.156)
Xor -> first layer([-3.0562,1.281][4.5288,-8.6027][-8.6027,-8.4429])
       second layer([-1.6355, 7.9396,-7.6963])
	   
After training, the weight is much smaller. Since I ran on virtual machine so I didn't put too much loop for saving time.
I use 500 samples and ran 100 times(forward, backward, and updateParameters) 
If loop or sample increases, the weight would be much smaller.


##Description of code.

###(1)neuralNetwork.py
For running the code.
First, declare a constructor with 2 input, a hidden layer with 2 nuruon, and 2 output
ex: N = NeuralNetwork(2,2,2) #(number of input, number of neuron in the first hidden layer, ... , number of output)

Then, you can call the layer for changing the weight value
ex: theta = N.getLayer([])

For the input value, you should use pytorch's FLoatTensor and also design your output Target with pythorch's float tensor
dtype = torch.FloatTensor
forwardvect = torch.Tensor([0.05,0.1]).type(dtype)
backwardvect = torch.Tensor([0.01,0.99]).type(dtype)

Then do the forward pass, backward propogation, and update the weight
I = N.forward(forwardvect)
N.backward(backwardvect)
N.updateParams(0.5)

Note:
You can also run the 2D tensor
Just feed the 2D input and output
ex: N = NeuralNetwork(2,2,2)
arr = torch.randn(2,4).type(dtype)
I = N.forward(arr)

###(2) Logic gate
First create constructor
And = AND()

Then, train the data
And.train()

Finally, get the output
print "And(True, True) = ", And(True, True)
print "And(False, True) = ", And(False, True)






