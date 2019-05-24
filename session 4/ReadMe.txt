								Architectural Basics

1) Basics Concepts
	Kernels and how do we decide the number of kernels:

		A kernal is a nxn matric used to do convolution operation.
		For 400x400 image, 512 number of kernals are needed in the last layer of each convolution block.
		This is for all the natural images.
		
		Depending on the complexity of the dataset and the accuracy needed, required number of kernals will differ.
		We need to try and figure out how many number of kernals are required for a given dataset.
	
	3x3 Convolutions:
	
		3x3 are the most prefered kernals as the numer of parameters are less.
		There is also a sense of left and right with odd sized kernal rather than a kernal of even dimentions like 4x4.
	
	MaxPooling:

		Max pooling is not a convolution.
		When 2x2 MaxPooling operation is done on an input image, out of every adjecent 4 pixel values, max value is picked
		irrespective of where that value is location among the 4 adjecent pixels.
		There is a sense of quantization in terms of space and this leads to a slight translational and rotational invariance.
		
		This will be used in transion block to reduce the number of parameters.
		
		Most prefered size is MaxPooling(2x2).Resoltion decreases if we go for MaxPooling of higer dimention(like 3x3 etc).
		Receptive field doubles with MaxPooling (2x2).
		Resolution will become half after MaxPooling (2x2)

	1x1 Convolution:
		1x1 is used to select the filters.
		It selects all the kernals which detected various features of a class.
		This will be used to reduce the number of kernals typically.

	SoftMax:
		If the changes of an image getting classified as A,B and C are given by x,y and z respectively,
		and if softmax Activation function is applied then softmax values for each of these classes A,B and C are given by
		Softmax for A:  (e to the power x)/[(e to the power x)+(e to the power y)+(e to the power z)]
		Softmax for B:  (e to the power y)/[(e to the power x)+(e to the power y)+(e to the power z)]
		Softmax for C:  (e to the power z)/[(e to the power x)+(e to the power y)+(e to the power z)]
		
		For Ex: If the changes of an image getting classified as A,B and C are given by 5,4 and 2 respectively,
				then softmax values for each of these classes A,B and C are given by
				Softmax for A:  (e to the power 5)/[(e to the power 5)+(e to the power 4)+(e to the power 2)] = .705
				Softmax for B:  (e to the power 4)/[(e to the power 5)+(e to the power 4)+(e to the power 2)] = .259
				Softmax for C:  (e to the power 2)/[(e to the power 5)+(e to the power 4)+(e to the power 2)] = .035
		
	Image Normalization:
		For the images, the intensity of the pixels vary from 0 to 255.
		In image normalization, the intensity of each pixel is devided by 255( 255 - 0 ) to make sure that the value is in between 0 and 1.
		This wil ensure that on convolution, pixel with large intensity and pixel with very low intensity will not vary drastically.
	
	Position of MaxPooling:
		After all the edges and gradients are discovered, we can afford to introduce MaxPooling to reduce the number of trainable parameters.
		Typical network architecture is CTCT...CTC(C for Convolution Block and T for Transition Block).
		MaxPooling will be used in each of the transional block.

		MaxPooling should not be at the very begining of the network or towards the end (just before the flatten and softmax).
		Otherwise, accuracy will not impacted.
	
	Concept of Transition Layers:
		Transition block consists of 1x1 convolution followed by MaxPooling(typically 2x2).
		1x1 convolution will selects the related kernals from all the kernals in the convolution blocks.
		
		2x2 maxpooling will reduce the image resolution to half which doubling the receptive field.
		This helps reduce the trainable parameters and introduces slight translational and rotational invariance.
		
		
	Position of Transition Layer:
		Aboriginy network architecture consisted of a series of convolutions with kernals of various sizes.
		Later, it is being realized that convolutions with just 3x3 kernals alone will give the same accuracy for less number of trainable parameters.

		The subsequent improvement in network architecture is CTCT...CTC(C for Convolution Block and T for Transition Block).
		Convoltion block consisted of a series of 3x3 convoltions till the block is expressive enougth to capture all the features.
		
		Transition block consists of 1x1 convolution followed by MaxPooling.
		



2)Advanced concepts:
	Receptive Field:
		Receptive field represents the number of pixels seen by any pixel at a spectific larger.

		After a 3x3 convolution, each pixel in the resulant image has seen 3 pixels. Hence the receptive field at that level is 3.
		
		After two times of 3x3 convolution, each pixel in the resulant image has seen 5 pixels. Hence the receptive field at that level is 5.
		
		After three times of 3x3 convolution, each pixel in the resulant image has seen 7 pixels. Hence the receptive field at that level is 7.
		
		
	How many layers,
		We need to have neural network layers till the global receptive field is equal to the size of the object to be detected from the image.
		
	When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
		When the resoltion of image reached 7x7, convoltion with 3x3 will not cover all the pixels in the images equal number times.
		Hence, at that is needed at that level is a convolution with 7x7 kernal.



3)Training the network
	Learning Rate:
		Learning rate is a parameter which defines how fast the traing curve converges.
		A very low learning rate takes large number of cycles to coverge.

		Very large learning rate will cause the trainging cover to overshoot the coverges poing and it keeps oscillating about the covergence point.
		It will not coverge at all.

		We need to have a moderate learning for reasonably fast convergence.
		We need to adopt and  keep reducing the learning rate as we move closer to the convergence point.
		
	Adam vs SGD:
		Adaptive moment estimation (Adam) and Stochastic Gradient Descent (SGD) are coule of optimization algorithems 
		used in machine learning to optimize the error rate.
		
		Speed of converge and performance of model on new data are the metrics that define the efficiency of the optimizers.
		
		SGD performs computatios on random and smaller subset of data.SGD produces same performance when the learning rate is low.
		
		Adam combines the advantages of two SGD extensions and computes individual adaptive learning rates for different parameters.
		SGD extenstions are	
			Root Mean Square Propagation (RMSProp) and 
			Adaptive Gradient Algorithm (AdaGrad)

		
	Number of Epochs and when to increase them:
		As long as the training accuracy and validation accuracy are improving and there is a difference between there, 
		the number of epoches can be increased.


4)Improving accuracy during Training

	Batch Normalization:
		Batch Normalization does normalization accross the kernals of a channel.
		Batch Normalization should be added before every convolution layer.
		It should be avoided before prediction

	DropOut:
		When drop out is used, randomly the output of some of the neurons are droped from being sent to the next layer.

	When do we introduce DropOut, or when do we know we have some overfitting:
		When the network is doing good on test dataset in terms of accuracy and poor accuracy on validation data set,  
		this is being refered to as overfitting.
		
		Introducing dropout will help overcome overfitting.
		After every larger, dropout layer can be added with a value like 0.1
		In the transtion block, the drop out value can be more like 0.2 but dropout should be avoided after the last convoltion layer.
	
	The distance of MaxPooling from Prediction:
		MaxPooling should not be used in the last Transition block(1x1 conv) and after the last convolution layer (i.e before flatten and softmax).
		If used, it will reduce the accuracy.

	The distance of Batch Normalization from Prediction:
		Batch Normalization should not be used in the last Transition block(1x1 conv) and after the last convolution layer (i.e before flatten and softmax).
		If used, it will reduce the accuracy.

	How do we know our network is not going well, comparatively, very early
		Comaparing the accuracy of the first two epoches, we can tell how well the network is doing comparitively.
	
	Batch Size, and effects of batch size:
		The validation accuracy of the network will improve till a particular value and beyond that the validation accuracy decreases.
		
	When to add validation checks:
		Validation check can be added while training by adding validation_data=(X_test, Y_test) as below
		
		model.fit(X_train, Y_train, batch_size=128, nb_epoch=90, verbose=1, validation_data=(X_test, Y_test))
	
	LR schedule and concept behind it:
		During trainging, as the accuracy gets closer to the convergence point, it is good to reduce the learning rate so that
		the accuracy doesn't overshoot the convergance point.
		
		The learning rate can be reduced in a predefined schedule.
		Some of the commonly used learning rate schedules are 
		i)Time-based decay 
			Mathematically,  lr = lr0/(1+kt)  where lr0 is the previous learning rate, k is a hyper parameter and t is the number of iteration. 
							
		ii)Step decay and
			Mathematically, lr = lr0 * drop^floor(epoch / epochs_drop)
			Typical drop rate is drop by half for every 10 epoches.
			Hence drop = 0.5 and epochs_drop = 10.0

		iii)Exponential decay
			Mathematically, lr = lr0 * e^(−kt), where lr0 is the previous learning rate, k are hyperparameters and t is the iteration number.
