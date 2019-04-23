# Code for machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import py_vision.image_processing.image_processing as im_proc
except:
    import sys
    sys.path.append('C:\ken\GitHub\CampbellMuscleLab\Projects\Python_MyoVision\kens_tinkering\Python_code')
    import image_processing.image_proc as im_proc


def learn_test_1(excel_file_string,
                 output_classifier_file_string=""):
    # Testing learning - based on
    # https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix
    

    # Load data
    d = pd.read_excel(excel_file_string)
    
    # Divide the data into predictors (X) and classifications (y)
    # y=1 - fiber, 2-connected, 3-intersitital, 4-other
    X = d.drop(['label','classification'], axis=1)
    y = d['classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    # Train
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)

    # Prediction
    y_pred = svclassifier.predict(X_test)

    # Evaluate
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    if (output_classifier_file_string):
        import pickle
        pickle.dump(svclassifier,
                    open(output_classifier_file_string, 'wb'))

def classify_labeled_image(im_label, classifier_model):
    # Code implements the classifer on a labled image
    
    # Get the blob data
    blob_data, region = im_proc.calculate_blob_properties(im_label)

    # Apply the classifer
    X = blob_data.drop(['label'], axis=1)
    c = classifier_model.predict(X)

    # Add the prediction column
    blob_data['predicted_class'] = c

    # Create a new image showing the classification
    im_class = np.zeros(im_label.shape)
    for i in np.arange(len(blob_data)):
        im_class[im_label==(i+1)] = c[i]

    # Return useful stuff
    return im_class, blob_data

def load_classifier_from_file(classifier_file_string):
    # Loads in a classifier from a pick file

    import pickle
    svclassifier = pickle.load(open(classifier_file_string, 'rb'))

    return svclassifier

def implement_classifier(raw_image_file_string, classifier_file_string):
    # Code uses a prior model to predict features

    from skimage.color import label2rgb

    # Turn raw_image_file_string into a labeled image
    im_label, im_sat = \
        im_proc.raw_image_file_to_labeled_image(raw_image_file_string)

    # Get the blob data
    blob_data, region = im_proc.calculate_blob_properties(im_label)

    # Load the classifier
    classifier_model = load_classifier_from_file(classifier_file_string)

#    # Implement the classifier
    im_class, blob_data = \
        classify_labeled_image(im_label, classifier_model)

    # Deal with potentially connected fibers
    im_class2, im_label2 = \
        im_proc.handle_potentially_connected_fibers(im_class, im_label,
                                                    blob_data, region,
                                                    classifier_model)

    # Shuffle im_label for improved display
    im_shuffle = im_proc.shuffle_labeled_image(im_label)
    im_shuffle2 = im_proc.shuffle_labeled_image(im_label2)

    fig, ax = plt.subplots(3, 2, figsize=(10,10))
    ax[0, 0].imshow(im_shuffle)
    ax[0, 1].imshow(im_class)
    ax[1, 0].imshow(im_label2)
    ax[2, 0].imshow(im_shuffle2)
    ax[2, 1].imshow(im_class2)

#
#    # Create overlay
#    im_overlay = label2rgb(im_class, im_sat)
#
#    
#    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), nrows=2)
#    ax1.imshow(im_overlay)
#    ax2.imshow(im_shuffle)

    # Return useful stuff
#    return im_mask, im_label, blob_data, region
