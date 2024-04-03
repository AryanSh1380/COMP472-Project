# COMP472 Project
## Team AK_6
### Aryan Shirazi – 40119594 – Role: Evaluation Specialist 
### Adnan Saab – 40075504 – Role: Training Specialist 
### Karim Tabbara – 40157871 – Role: Data Specialist 


- DataSet folder: Contains 25 images from each category (Neutral, Focused, Surprised, Happy) + Contains document with information about publicly available dataset used /

- Signed Expectations of Originality Forms: Contains signed forms from each of the three members of the group

- Python Code: 
"Assignment1.py" --> Contains code for reading data from the csv file containing image paths and labels + Contains code for computing the class distribution + Contains code for displaying a sample of the images from each class along with their pixel intensity distribution

"Ass2_mainModel.py" --> Contains code for creating and training the main CNN model 

"Ass2_modelVar1.py" --> Contains code for creating and training the variant 1 CNN model 

"Ass2_modelVar2.py" --> Contains code for creating and training the variant 2 CNN model 

"modelEvaluation.py" --> Contains code for evaluating main model 

"modelEvaluation_Var1" --> Contains code for evaluating variant 1 

"modelEvalation_Var2" --> Contains code for evaluating variant 2 

"picEval.py" --> Contains code for testing the model for 1 image 


- CNN Models: Main Model --> main_model.pt  +  Variant 1 --> modelVar1.pt  +  Variant 2 --> modelVar2.pt 

- README.txt (this file) 


HOW TO RUN OUR CODE FROM ASSIGNMENT 1:

1) Clone the github repo (if using Git Bash, use the following command:
git clone "https://github.com/AryanSh1380/COMP472-Project"

2) Make sure that each of the Neutral, Focused, Surprised, Happy folders contain images

3) Run the code "Assignment1.py"

4) Wait until Figures appear on the screen

5) The figures that will appear will correspond to:
Figure 1 - Class Distribution Bar Chart
Figure 2 - 25 Neutral Images
Figure 3 - Pixel Intensity Distribution of the 25 Neutral Images in Figure 2
Figure 4 - 25 Focused Images
Figure 5 - Pixel Intensity Distribution of the 25 Focused Images in Figure 4
Figure 6 - 25 Surprised Images
Figure 7 - Pixel Intensity Distribution of the 25 Surprised Images in Figure 6
Figure 8 - 25 Happy Images
Figure 9 - Pixel Intensity Distribution of the 25 Happy Images in Figure 8




HOW TO RUN OUR CODE FROM ASSIGNMENT 2: (the program could take a while to run)
Need to do the first 2 steps from assignment 1, just to make sure the repo is cloned and the dataset is properly loaded into the project.
Cuda can be used to accelerate the running of the code

- Training
1) Select the python code file corresponding to the model you want to train
2) Run the code (the output should show the loss, followed by the accuracy on the test set after the epoch, followed by the best accuracy for all epochs so far)


- Image Classification
1) In "picEval.py", on line 63, optionally replace 'Happy/ffhq_30.png' with any path to a picture (Make sure the path to the test image chosen is correct and exists)
2) Depening on which of our models you want to test be sure to uncomment that import file on lines 6 to 9.
3) Uncomment the model you want to use on line 58 and comment out the two other models.
4) Run the code and evaluate the output (it will return the name of the predicted class).


- Model Evaluation
1) Choose the python file corresponding to the CNN model you would like to evaluate
To evaluate Main Model --> modelEvaluation.py
To evaluate Variant 1 --> modelEvaluation_Var1.py
To evaluate Variant 2 --> modelEvaluation_Var2.py
2) Run the code and evaluate the output (accuracy, macro and micro precision/recall/f1 measure) + see confusion matrix plot

