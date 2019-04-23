Contents:	README.txt:			project documentation
		MyoVision_source
			MyoVision.py:		main for program
			MuscleStain.py:		file that describes class to processes muscle stain
		functions
			display_mult_ims.py:	general purpose functions for looking at images
			im_overlay.py:		overlays mask on image
			image_processing.py:	functions based on MATLAB image processing functions
			lists2txt.py:		function to write numpy ndarray into txt file
			pickle_helpers.py:	wrappers for pickling objects
			sklearn_helpers.py:	functions to ready data for sklearn SVMs
		images
			folder of test images

Description:
	Use this program for processing of muscle stain images

Implementation Notes:
	

Running:
	-Open Windows Command Prompt and navigate to the MyoVision_source folder
	-Activate the virtual environment with:
		venv\Scripts\activate
	-run the MyoVision program:
		python MyoVision.py your_classifier_input.xml input_image_folder\\ output_data_folder\\
			--Where you use the following parameters:
			-your_classifier_input.xml - a file specified in the Classifier input section of this document
			-input_image_folder\\ - relative path to folder with images you wish to process
			-output_data_folder\\ - relative path to folder where you wish to put the output
	-Example run:
		python MyoVision.py input.xml images\\ output\\


Classifier input:
	The input file for the classifier is an XML document describing which Excel files to pull metric and classification data from.
	This file consists of 2 important sections:
	-classification_files:
		-The Excel documents specified in this section will point to manually classified data in a format matching that in the example documents
		-Add further classification files by typing the following in the <classification_files> section:
			<c#>...</c#>
			--Where:
				-# is the next integer after the previous entry (starting at 0 if first file)
				-... is relative path to needed Excel file
	-metric_files:
		-The Excel documents in this section will point to shape descriptor data corresponding to the manual classifications in the same-numbered classification_file document
		-You must have a matching metric_file for each classification_file
		-Add further metric files by typing the following in the <metric_files> section:
			<m#>...</m#>
			--Where:
				-# is the next integer after the previous entry (starting at 0 if first file)
				-... is relative path to needed Excel file


Programming with MuscleStains:
	See the using_MuscleStains.txt document for more detailed description of using the MuscleStain class in MyoVision

References:
	