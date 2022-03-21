INTERPRETING AND TRANSLATING SIGN LANGUAGE                                                   








Interpreting and Translating Sign Language Using Machine Learning

Nevin Gregory & Justin Moonjeli

Gwinnett School of Mathematics, Science, and Technology

# **Literature Review**
## **What is Machine Learning?** 
Machine Learning is a relatively new technology, first proposed “in 1944 by Warren McCullough and Walter Pitts, two University of Chicago researchers who moved to MIT in 1952” (Hardesty, 2017). After that from about 1949 to the late 1960s “[Arthur Samuel] did the best work in making computers learn from their experience… [using a] game of checkers” (McCarthy & Feigenbaum, n.d.). It is a learning algorithm that allows a computer to take a training set and find patterns with which it can detect and interpret similar samples. It is a revolutionary type of learning algorithm that does not plateau as more data is fed in (Brownlee, 2019). This is the best type of algorithm to use for this particular project, as instead of having to code the detection algorithm ourselves, the computer can begin to “learn” like a human and detect the sign language symbols on its own. Because of this, working models can be produced quicker than usual and more efficiently.
## **Training, Testing, and Validation Sets**
Al-Masri’s (2018) article discussed training, validation, and test data sets in machine learning and showed that the three previously stated factors are critical to achieving a well functioning model. The author emphasized the importance of the model examining data, the model leaning from its fallacies, and the conclusion on how well it performed (Al-Masri, 2018). Al-Masri explains what the three key factors are and why they are vital in creating an efficient and functional model. The training set is what the model learns of off by adjusting various parameters (Al-Masri, 2018). The validation set is used to periodically evaluate the model, as it would be impossible to train the program without assessing it and finding an error rate (Al-Masri, 2018). This is the crucial portion of training as the results of these assessments are what the model will base is newly tuned parameters off. The test set’s purpose is to evaluate the model after its completion of the training phase (Al-Masri, 2018). This is a critical phase as it is what decides whether or not the model is ready for application or if it needs more training and validation data.
## **Parts of a Convolutional Neural Network:**
**Structure of a convolution neural network.** Convolution Neural Networks, CNN, contain various parts to help with training and with outputting inferences and relations based on the inputted data. A CNN has four main parts, convolution layers, rectified linear unit layers, pooling layers, and a final or semifinal fully connected layer (Bonner, 2019). The first part of a CNN is convolution. The primary focus of convolution is to extract key features from an image (Bonner, 2019). This is done by using filters, looking for specific features within a receptive field. The importance of features and distinguishing factors are called weights and can be adjusted depending upon the needs of the system (Neilsen, 2019). After a series of filters are applied, a feature map is created, which is an extremely condensed form of the initial image (Bonner, 2019). These processed layers typically focus on specific elements depending on the iteration. The first layer detects various gradients in the image, the second layer targets lines, and the third layer looks at shapes (Tch, 2019). This pattern of increasing complexity continues until required to stop by a preset parameter. The rectified linear unit layer is an addition to the convolution layer. Its main task is to increase non-linearity in the network (Bonner, 2019). This is done because convolution is quite linear, whereas, in real life, things are very nonlinear. Using the ReLU layer can thus help the network train faster all while avoiding hits to generalization accuracy. The next part is pooling, which is where key features are boosted while less important ones are removed (Bonner, 2019). The final part is the fully connected layer in which the various neurons in the network chose a class for the image based on the data inputted after the flattening off of predetermined weights (Loy, 2018). The setting of these weights is via numerous training and validation permutations (Montana & Davis, n.d.). Many CNN’s are trained using datasets, large libraries of images that are pre-partitioned based off of class (Olafenwa, 2018). These data sets are often separated into two main groups, the training set and testing set (Google Developers, n.d.).

**The Google Colab environment.** The calculations that are needed for machine learning are best accomplished by a Graphics Processing Unit (GPU). Since these calculations are quite intensive, it takes a powerful GPU to perform it. Fortunately, Google has released an environment called Google Colab. This platform provides a free environment that uses free NVIDIA K80 12GB GPUs given by Google. In addition to this, Google Colab will resolve “installation issues” and “[creates a] local environment with anaconda” (Mandal, n.d.). Setting up the environment for this experiment is simple. First, one must create a new notebook that uses the Python language. Then the GPU must be selected as the runtime type. From there, the process is very similar to what one would use when doing it on the local computer. Pip is the package manager used to install new Python libraries, and notebooks can be downloaded to be used for other purposes (Mandal, n.d.). This platform provides a quick, easy way to set up an environment for machine learning that will speed up the programming process significantly.

## **Other Parts of the Program**
In order to fully deploy a prototype for the program, it needs to be able to do two additional tasks that do not include machine learning. There must be a way to get a video of the sign language. However, the machine learning model does not accept videos. Therefore, the video itself must be cut down into separate frames that would function as the images to be passed in. The next step is to be able to communicate with the deaf person naturally. A good way for this is a speech recognition program. Since these two items are not the main focus of the project, and there is no need to recreate them, libraries can be used, which will assist in the timely completion of these tasks.

**Getting input from the Webcam.** The first goal is to split the webcam video into still frames which can then be passed through a neural network model. To save time while doing this, a good method is to use the popular computer vision library OpenCV. There are three important functions that make up this relatively simple program. These functions are (1) the VideoCapture(File\_path) function, which “ Read[s] the video(.mp4 format),” (2) the read() function, which “Read[s] data depending upon the type of object that calls,” and (3) the imwrite(filename, img[, params]) function, which “Saves an image to a specified file” (Python | Program to extract frames using OpenCV, n.d.). Simply explained, the code defines a function through which an mp4 file is passed (in this case it would be the webcam stream). It then calls the VideoCapture function which returns a VideoCapture object that is now stored in a variable. Two more variables are defined, one to keep track of how many frames have gone by and one to keep track of whether or not there are frames left to check. Then a loop is run which continues until there are no more frames. The read() function is called on the VideoCapture object and is stored in the variable which sees if there are remaining frames and a new variable that will store the image of the frame. Lastly, the imwrite() function is called, which will store the frame into a specified path. This final image is what will be passed into the neural network as the input.
**Speech recognition.** While the script is relatively simple, the procedure for converting speech to text is a little more complicated. The basic idea is surprisingly similar to the technique used for the webcam. This model is called the “Hidden Markov Model (HMM)” (Amos, n.d.). The way the model works is by “[dividing] the speech signal… into 10-millisecond fragments… The final output of the HMM is a sequence of these vectors” (Amos, n.d.). These short vector fragments can be used in speech recognition. There are several libraries in use with this program. One is the Recognizer class. This is used for the actual recognition of speech. The next is the Microphone class, which is what is used to receive input from the system or external microphone. Once it receives audio from the microphone, it must be passed into the Recognizer class, which will interpret the sound file. Of course, the microphone input is not always going to be one hundred percent perfect quality. There is going to be ambient sound. Luckily, there is also a function that accounts for ambient sound. After all of these calculations, the class will translate the sound into text, and from there, the text can be taken and displayed on a screen.
#
# **Statement of Purpose**
The objective is to create a lightweight prototype program that by using a neural network model trained with American Sign Language (ASL), can interpret and translate hand articulations collected and recorded by a camera. The system will then display the American English translation of the actions performed. This will be made to serve the needs of those with speaking handicaps by helping them communicate more effectively and naturally with others unfamiliar with ASL. 
# **Hypotheses**
**Hypothesis**

The team believes that the neural network, after rigorous training and sufficient convolving, will correctly be able to translate the image snapshots inputs of a subject using American Sign Language into American English and be displayed on the program running system, reaching an accuracy of at least 95% by epoch 2000.

**Alternative Hypothesis**
**
The neural network, after training and sufficient convolving, will incorrectly or ineffectively translate the image snapshots inputs of a subject using American Sign Language into American English.

**Null Hypothesis**

The neural network will not have the desired effect on the translation of American Sign language into American English.


**Materials:**

- 1 Webcam (Internal or External)
- 1 Windows Laptop or Desktop Computer
- The Google Colab Environment
- 1 High Thread Count Capable Graphics Card (NVIDIA K80 provided by Google via Colab)
- American Sign Language Dataset

**Part 1: Constructing the Neural Network**

1. Acquire a dataset containing various hand symbols from American Sign Language (ASL)
1. Split the dataset into training and testing sets.
1. Construct an image recognition neural network and train it with the training set which was taken from the original dataset
   1. The language used is Python 3 within the Google Colaboratory API
   1. The package used is TensorFlow, an open source platform for machine learning.
   1. The neural network accepts a 28x28x1 grayscale image, contains 2 5x5 Convolutional layers, a Pooling layer with a stride of 2 pixels, and a fully connected layer. The activation function used is ReLU.
1. Run the trained network through the testing set to determine the accuracy
1. Adjust the weights and biases
1. Repeat steps 3-5 until the network is at sufficiently high accuracy (95-100%).

**Risk and Safety**: Identify any potential risks and safety precautions needed. 

N/A. Due to the vast majority of the experimentation being digital, little to no risks are involved in this project.

**Variables**

Independent Variable - Epoch Number

Dependent Variable - Accuracy (%)







**Data Table**

|Epoch Number|Cost (bits)|
| :-: | :-: |
|0|0|
|100|1242155913.625|
|200|27972437.625|
|300|8719844.5|
|400|8057232.25|
|500|4141510.21875|
|600|1945596.90625|
|700|1859662.7890625|
|800|1744056.421875|
|900|1172214.8437|
|1000|269439.09375|
|1100|660847.59375|
|1200|0.0|
|1300|342277.171876|
|1400|547804.90625|
|1500|374965.1875|
|1600|58576.7421875|
|1700|0.0|
|1800|0.0|
|1900|74830.875|
|2000|14393.75|



|Epoch Number|Training Accuracy (%)|Testing Accuracy (%)|
| :-: | :-: | :-: |
|0|0|0|
|100|34.375|42.96875|
|200|68.75|63.28125|
|300|78.90625|75|
|400|75|82.03125`|
|500|83.59375|88.28125|
|600|89.84375|93.75|
|700|90.625|92.1875|
|800|89.0625|96.09375|
|900|92.1875|98.4375|
|1000|97.65625|100|
|1100|98.4375|100|
|1200|100|100|
|1300|97.65625|98.4375|
|1400|95.3125|100|
|1500|99.21875|100|
|1600|99.21875|100|
|1700|100|99.21875|
|1800|100|100|
|1900|98.4375|100|
|2000|99.21875|99.21875|




**Analysis**

![Chart](graph.png)



**T-test**


||*Variable 1*|*Variable 2*|
| :- | :-: | :-: |
|Mean|89.375|91.44531|
|Variance|251.362|228.6232|
|Observations|20|20|
|Pooled Variance|239.9926||
|Hypothesized Mean Difference|0||
|df|38||
|t Stat|-0.42261||
|P(T<=t) one-tail|0.337482||
|t Critical one-tail|1.685954||
|P(T<=t) two-tail|0.674964||
|t Critical two-tail|2.024394||



**Discussion**

As seen in both the graph and the data tables, as the epoch numbers increased, the accuracy of the neural network increased. At epoch zero, the accuracy was only around 34%. However, at the final epoch, 19, the neural network had reached an accuracy of 99%, which is exceptionally high. Using the accuracy data from the testing set, a t-test was performed. The results showed that the t-Stat value (-0.46621) was less than the t-Critical two tail value (2.024394). This meant that any difference was due to random chance. This proves our hypothesis as our neural network successfully outputted the correct letter of multiple sign language symbols that we passed in ourselves and had successfully trained. In addition to this, the cost function is also displayed. As can be seen, the cost decreased exponentially as the epochs increased, starting at around 124215591 bits and ending at around 14393 bits. This also helps confirm the hypothesis that was put forward.

These are the expected results, as that is the intended result of a neural network. As the network trains on a set of data and learns, then it will be better prepared for the next set of data and the next, and this causes the accuracy to rise. The results of the cost function were also expected as it should be the inverse of the accuracy function due to it measuring the error of the neural network. This is the trend followed by all strong neural networks.

A possible problem area for our project is overtraining. Overtraining is when a neural network is trained too much on a specific set of data, and instead of learning with a regular algorithm, it “memorizes” the data, which leads to inflation of accuracy that does not correctly show the capabilities of the network. The way to test for overtraining is to use outside data, data not directly trained with, to test to see if a sudden drop in accuracy occurs, which would be the next step in building the app. Another possible problem is a lack of data. Neural networks will learn better when given more data. This is why a larger dataset is more beneficial to accuracy in the completed model. This issue should not present too much of a problem because our training set was at a number that is generally thought of as an appropriate size for training (27,455 cases).

If the project was to be redone, the project team would like to use more time management tools such as the agile project management tool. Many software teams already use this type of project management. Using the quick scrimmage sessions, the project’s efficiency would be increased which would result in more work being accomplished. The research that we carried out can be used to produce a program to increase the ease of online communication for those with hearing and speech impairments. Using the trained neural network and using more datasets with more words, sign language can be a viable way to talk online. 



**Conclusion**

Through many iterations and evolutions, the created neural network confirmed the hypothesis, accurately translating the image snapshot inputs of a subject using American Sign Language into English and display the text on the program running system. After the successful completion of 2000 epochs, an accuracy of 99% was achieved, showing the capabilities of the program. The peak accuracy value was 100%, making it nearly perfect in translating ASL into English. This high value shows the flexibility of neural networks, and in specific, convolutional neural networks and how they can take the input data, and with minimal losses. The T-test run on the testing data resulted in the t-Stat value (-0.46621) being less than the t-Critical two-tail value (2.024394), which shows that the effectiveness of the network comes from its proper training rather than random chance. The neural network trained will be made market-ready through the implementation of a clean and easy to use GUI for mobile or PC application, and the optimization of weights and biases. The neural network and the program it will fit into can become a valuable tool in many fields including but not limited to business communication, basic HTH, human to human, interactions, interdisciplinary exchange, education applications, and as a tool to better the lives of those who are reliant on speaking ASL to others.

**Acknowledgments**

We would like to thank Dr. Susan Thomas and Mr. William Castle for providing us with inspiration for the project. We would also like to thank Dr. Gregory Job and Mr. Arun Kanjira for providing us with the necessary resources and tools to create and develop the prototype.





#
# **References**
Al-Masri, A. (2018, December 22). What are training, validation and test data sets in machine learning? Retrieved September 22, 2019, from Medium website: <https://medium.com/datadriveninvestor/what-are-training-validation-and-test-data-sets-in-machine-learning-d1dd1ab09bae>

Amos, D. The ultimate guide to speech recognition with python. Retrieved from <https://realpython.com/python-speech-recognition/>

Bonner, A. (2019, June 1). The complete beginner’s guide to deep learning: Convolutional neural networks and image classification. Retrieved August 21, 2019, from Medium website: <https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb>

Brownlee, J. (2019, August 15). Deep learning & artificial neural networks. Retrieved September 22, 2019, from Machine Learning Mastery website: <https://machinelearningmastery.com/what-is-deep-learning/>

Hardesty, L. (2017, April 14). Explained: Neural networks. Retrieved September 22, 2019, from MIT News website: <http://news.mit.edu/2017/explained-neural-networks-deep-learning-0414>

Loy, J. (2019, March 29). How to build your own Neural Network from scratch in Python. Retrieved September 22, 2019, from Medium website: <https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6>

Mandal, S. (2019, May 1). How to use google collab. Retrieved September 22, 2019, from GeeksforGeeks website: <https://www.geeksforgeeks.org/how-to-use-google-colab/>

McCarthy, J. & Feigenbaum, E. (n.d.). Arthur Samuel: Pioneer in machine learning. Retrieved September 22, 2019, from <http://infolab.stanford.edu/pub/voy/museum/samuel.html>

Montana, D. & Davis, L. (n.d.). Training feedforward neural networks using genetic algorithms. Retrieved September 22, 2019, from <https://www.ijcai.org/Proceedings/89-1/Papers/122.pdf>

Nielsen, M. A. (2015). Neural networks and deep learning. Retrieved from <http://neuralnetworksanddeeplearning.com/chap1.html>

Olafenwa, M. (2018, July 26). Train Image Recognition AI with 5 lines of code. Retrieved September 22, 2019, from Medium website: <https://towardsdatascience.com/train-image-recognition-ai-with-5-lines-of-code-8ed0bdd8d9ba>

Python | Program to extract frames using OpenCV. (2018, May 15). Retrieved September 22, 2019, from GeeksforGeeks website: <https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/>

Tch, A. (2017, August 4). The mostly complete chart of neural networks, explained. Retrieved September 22, 2019, from Medium website: <https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464>

Training and test sets: Splitting data | machine learning crash course. (n.d.). Retrieved September 22, 2019, from Google Developers website: <https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data>
