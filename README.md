# ITE@UIUC Engineering Open House 2023

## Contributors:
- #### Louis Sungwoo Cho (Civil & Environmental Engineering Transportation & Computer Science Minor)
- #### Joey Junyi Cao (Aerospace Engineering)
- #### Saranya Yegappan (Civil & Environmental Engineering)
- #### Brandon Tomic (Civil & Environmental Engineering Transportation)
- #### Jesse Ekanya (Civil & Environmental Engineering Transportation)
- #### Sam Guagliardo (Civil & Environmental Engineering Transportation)
- #### Ethan Chow (Computer Engineering)

## Project Description:
Institute of Transportation Engineers UIUC Chapter (aka ITE@UIUC) is a registered student organization at the University of Illinois at Urbana-Champaign. Our organization has been working on collecting images of transit buses operated by the Champaign-Urbana Mass Transit District (CUMTD). 


## Deep Learning Model Motivation
South Korea first opened their high-speed rail network on April 1st, 2004 to make rail travel time more fast and convenient. When I first traveled to South Korea, I still remember when I took KTX with my family for the first time when we went to Busan. I was excited to ride the high-speed train because the U.S does not have bullet trains which can travel as fast as the KTX trains. After nearly 2 decades the first KTX line the Gyeongbu High-Speed Line (경부고속선) connecting Seoul to Busan opened, the high-speed rail network has expanded almost throughout the entire country including the Honam High-Speed Line (호남고속선) connecting Seoul to Gwangjusongjeong to Mokpo, Suseo High-Speed Line or Sudogwon High-Speed Line (수서고속선/수도권고속선) connecting the south side of Seoul Suseo to Busan to Gwangju to Mokpo, Gyeongjeon Line (경전선) connecting Seoul to Masan to Jinju, Jeolla Line (전라선) connecting Seoul to Yeosu-EXPO, Donghae Line (동해선) connecting Seoul to Pohang, Gangneung Line (강릉선) also known as the 2018 Pyeongchang Olympics Line connecting Seoul to Gangneung, Yeongdong Line (영동선) connecting Seoul to Donghae, Jungang Line (중앙선) connecting Seoul to Andong (sections to Uiseong, Yeongcheon, Singyeongju, Taehwagang, Busan-Bujeon to be opened in December 2023), and the Jungbunaeryuk Line (중부내륙선) connecting Bubal to Chungju. As seen above, due to the continuing expansion of the South Korean high-speed train network, Hyundai ROTEM has designed many different types of variants to serve in various lines depending on their operational speed respectively. Due to each locomotive having unique features, I decided to create a deep learning model that can classify the 8 types of trains: KTX-1, KTX-EUM, KTX-Sancheon, and SRT. 



# Image Preparation
Random CUMTD bus image datasets were used to train the neural network model for image classification. 158 files were then split into 8 categories with each category having 10 images of the same class. 

![title](images/random_ktx_one.png)
### Figure 1. above shows the 10 random KTX-1 images from the given image dataset.

![title](images/random_ktx_eum.png)
### Figure 2. above shows the 10 random KTX-EUM images from the given image dataset.

![title](images/random_ktx_sancheon.png)
### Figure 3. above shows the 10 random KTX-Sancheon images from the given image dataset.

![title](images/random_srt.png)
### Figure 4. above shows the 10 random SRT images from the given image dataset.

Once all the random image datasets were printed out, the entire image dataset was split into training and testing sets. 80% of the total image datasets were used for training and the remaining 20% of the total image datasets were used for testing. The epochs number was set to 20 so the training model was run for 20 times. Then all the data were shuffled before the neural network model was created. 

# Convolutional Neural Network (CNN) Model
Convolutional Neural Network (CNN) model was used to classify the high-speed train images. One of the biggest advantage of using CNN models is that the neural network is able to detect the important features into several distinct classes from the given image datasets without any human supervision and also being much more accurate and computationally efficient. Hence, this deep learning model was chosen to train all the bus image datasets for this project. 

![title](images/cnn_process.png)
#### Figure 5. above shows how the cnn model processes the image dataset with series of convolution and pooling before flattening out the image to predict the output.

The model used for this project performs multiclass classification so the output is set to be softmax. But why is convolution so crucial in image classification? Convolution is a set of mathematical operations performed by the computer to merge two pieces of critical information from the image. A feature map for the images is produced using a 'convolution filter'. 

![title](images/cnn_filter.png)
#### Figure 6. above shows how the convolution filter produces the feature map.

The convolution operation is then performed by splitting the 3 by 3 matrix into merged 3 by 3 matrix by doing an element-wise matrix multiplication and summing the total values. 

![title](images/cnn_matrix.gif)
#### Figure 7. above shows the matrix operation of the convolution filter.

![title](images/cnn_visual.gif)
#### Figure 8. above shows the visualization of the  convolution input of the image.

Once all the convolution has been performed on the image datasets, pooling is then used to reduce the dimensions, a crucial step to enable reducing the number of parameters shortening the training time and preventing overfitting. Maximum pooling was used for this model which only uses the maximum value from the pooling window. 

![title](images/cnn_pooling2d.png)
#### Figure 9. above shows the pooling of the processed image in a 2 by 2 window.

![title](images/cnn_pooling3d.png)
#### Figure 10. above shows the pooling of the processed image in a 3 by 3 window.

Finally after adding all the convolution and pooling layers, the entire 3D tensor is flatten out to be a 1D vector into a fully connected layer to produce the output. 
![title](images/cnn_imp.png)
#### Figure 11. above shows the visual implementation of the CNN model. 

##### Original Source for the CNN Explanation: [Towarddatascience Applied Deep Learning](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2#:~:text=The%20main%20advantage%20of%20CNN,CNN%20is%20also%20computationally%20efficient)

# Results
Once the CNN model was built for image classification training with a given number of training steps also known as epochs set to 20, the accuracy score graph and the loss score graph with respect to each epoch step were plotted. 

![title](images/accuracy.png)
#### Figure 12. above shows the accuracy score of the CNN model with respect to the number of steps. 

![title](images/loss.png)
#### Figure 13. above shows the loss score of the CNN model with respect to the number of steps. 

According to the plots above, the train accuracy is very close to the testing accuracy as the number of epochs gradually increases. Overall, the model has produced a relatively high training accuracy. The number of losses meaning the error between the actual image and the predicted image decreases as more number of epochs are given into the model. This means that the chance of predicting a given image dataset accurately is very high. 

# Prediction
Once all the image datasets have been processed and the accuracy and loss score have been analyzed, a few set of images were given into the model to determine whether the model is accurate enough predicting the train type of a given image. Testing datasets were given into the model and the predictor plots the actual image and the predicted image with a confidence score respectively. 

![title](images/predicted_output.png)
#### Figure 14. above shows the predicted output of each image data given into our model with the train type and the confidence score for each image.

From the image above, it is clearly evident that the predictor estimates the train class very accurately. The confidence is also very high for each results meaning that the results have turned out very well. Overall, the model performed very well with all the bus image datasets.
