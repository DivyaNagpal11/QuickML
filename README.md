# About QuickML

Build a Cerner specific web application which will help speed up development and interpretation of machine learning ideas by automating repetitive tasks to help with exploratory data analysis, feature selection, algorithm comparisons and containerization of models. It has been divided 2 components based on structured and unstructured data (images).

Objectives of the Project

To develop a Web Application on Django which will support the following features

	a) Associates will be able to perform exploratory data analysis for their datasets and come up with the optimal feature selection and labels for the model.
 
	b) Associates will be able to validate multiple algorithms results in parallel automatically to identify the best algorithm suited for their use case.
 
	c) Associates will be able to dockerize their model which is ready for deployment. 

Modules in the Project:

a) Project Handling:

	
	a) Create Project:
	    This is used to create the project.There are two different types of the project that can be created .
	        1.Data Processing
	        2.Image Processing 
	b) Load Project:
	    This is used to load the project.
	    For working on data processing project a csv file should be uploaded.
	    For an image processing project a zip file should be uploaded which should contain the different categories/classes of images in different folders. 
	c) Delete Project:
	    This is used to delete the project.
	d) Undo:
	    This is used to Undo the previous step in the project.

b) Data Processing:


	a) Exploratory Data Analysis:
	
		1) Summary:
		    Used to view the summary of the data

		2) View Data:
		     Used to view all the data

		3) Visualization:
		    Used to view the data in different Graphs

		4) Data Cleaning:
		    Used to clean the data
		    Data Cleaning has the following Features
		    
			a) Missing Values:
			    Used to handle missing values in the data
			b) Normalization:
			    Used to Normalize the data between the given values
			c) Outlier Detection:
			    Used to find out the outliers in the data
			d) Delete Columns:
			    Used to delete columns from the data

		5) Normal Distribution:
		    Used to convert the given feature into a Normal Distribution

		6) Encoding:
		    Used to encode categorical features in the given data

		7) Sampling:
		    Used to perform Sampling on a given feature

		8) Binning:
		    Used to perform Binning on numerical features

	b) Development:
		1) Modeling:

			a) Classification:
			    Used to train and create Classification models using different Classification Algorithms and create Api-Url for the models.

			b) Clustering:
			    Used to train and create Clustering Models using different Clustering Algorithms and create Api-Url for the models

			c) Regression:
			    Used to train and create Regression Models using different Regression Algorithms and create Api-Url for the models

		2) Validation:
		
			a) Classification Validation:
			    Used to Validate the trained Classification model with user input

			b) Clustering Validation:
			    Used to Validate the trained Clustering model with user input

			c) Regression Validation:
			    Used to Validate the trained Regression model with user input

c) Image Processing:
    
    
    1) Visualization:
            Used for viewing the different categories of image uploaded.
    
    2) Modelling :
    
            a) Convolution:
                    Used for extracting the features from the image
            b) Max pooling:
                    Used for pooling the maximum value from the image arrays , thereby decreasing it's size and complexity.
            c) Dropout:
                    Used as a method for reguralization by dropping out few neurons and thus decreasing the size of the neural network.
            d) Flatten:
                    Used for flattening the array as a one dimensional array.
            e) Dense:
                    Used as a fully connected layer which is furthur used for getting the final output.
    
    3) Validating :
            Used for validating the the uploaded image.
