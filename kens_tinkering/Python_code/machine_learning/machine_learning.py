# Code for machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import py_vision.image_processing.image_processing as im_proc
except:
    import sys
    sys.path.append('C:\ken\GitHub\CampbellMuscleLab\Projects\Python_MyoVision\kens_tinkering\Python_code')
    import image_processing.image_processing as im_proc


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

def implement_classifier(raw_image_file_string, classifier_file_string):
    # Code uses a prior model to predict features

    import pickle
    from sklearn.svm import SVC
    from skimage.color import label2rgb

    # Turn raw_image_file_string into a labeled image
    im_label, im_sat = \
        im_proc.raw_image_file_to_labeled_image(raw_image_file_string)

    # Get the blob data
    blob_data = im_proc.calculate_blob_properties(im_label)
    X = blob_data.drop(['label'],axis=1);

    # Load in the classifier
    svclassifier = pickle.load(open(classifier_file_string,'rb'))


    c = svclassifier.predict(X)
    
    # Create a new mask showing the classification
    im_mask = np.zeros(im_label.shape)
    for i in np.arange(len(blob_data)):
        im_mask[im_label==(i+1)] = c[i]

    # Overlay
    im_overlay = label2rgb(im_mask, im_sat, alpha=0.3)
    
    # Shuffle im_label
    im_shuffle = np.random.permutation(im_label.max()+1)[im_label]
    
    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), nrows=2)
    ax1.imshow(im_overlay)
    ax2.imshow(im_shuffle)
    
