# **HW2 - Yun-Jou Lin 0027875836** 

## Concept Image for Forward pass
![Image1](https://github.com/Roselin/DeepLearning/blob/master/HW2/Concept.png)

In this case (above image), you should type the following thing to test the code  

N = NeuralNetwork(3,2,1)  
dtype = torch.DoubleTensor  
Input_vect = torch.Tensor([0.5, 0.1, 0.3]).type(dtype)  
I = N.forward(Input_vect)  

###Note: In the test.py file

There are three examples
(1) I used the example provided on the class   
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/  
    You can get the exactly same value  
    
(2) The input is a 2D DoubleTensor row = 2, column = 4   
    The output is printed out  
    
(3) The tests of And, Not, XOR logic gate are provided.  
    The output is printed out  
