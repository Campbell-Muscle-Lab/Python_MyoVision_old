# Code for machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.image_processing import image_proc as im_proc

def learn_test_2(input_data_file_string, output_classifier_file_string):
    
    from sklearn import svm, datasets
    
    # Load data
    d = pd.read_excel(input_data_file_string)
    
    c = d['classification']
    d = d.drop(['classification','label'], axis=1)
#    d = d.loc[:,['area','eccentricity']]
    print(d.head())
    
    
    print('Fitting model')
    classifier = svm.SVC(kernel='linear')
    classifier.fit(d, c)
    
    # Save model
    if (output_classifier_file_string):
        import pickle
        pickle.dump(classifier,
                    open(output_classifier_file_string, 'wb'))
    
    var_names = d.columns
    no_of_variables = len(var_names)
    
    print('Making figure')
    # Make a figure
    fig, ax = plt.subplots(no_of_variables, no_of_variables)
    for i in np.arange(0, no_of_variables):
        for j in np.arange(i, no_of_variables):
            x = d.iloc[:,i]
            y = d.iloc[:,j]
            
            x_step = 0.01 * (x.max() - x.min())
            y_step = 0.01 * (y.max() - y.min())

#            xm,ym = make_meshgrid(x,y,[x_step,y_step])
#            plot_contours(ax[i,j],classifier,xm,ym,
#                          cmap=plt.cm.coolwarm,s=20,edgecolors='k')
            
            ax[i, j].scatter(x, y, c = c)
            ax[i, j].set_xlabel(var_names[i])
            ax[i, j].set_ylabel(var_names[j])
            
            print('i=%d, j=%d' % (i,j))
    
def make_meshgrid(x, y, h=[1, 1]):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 10*h[0], x.max() + 10*h[0]
    y_min, y_max = y.min() - 10*h[1], y.max() + 10*h[1]
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h[0]),
                         np.arange(y_min, y_max, h[1]))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

    # Train
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)

    # Prediction
    y_pred = svclassifier.predict(X_test)
    print(y_pred.size)

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
    im_class = np.zeros(im_label.shape, dtype = 'uint8')
    for i, r in enumerate(region):
        if (np.mod(i + 1, 100) == 1):
            print('Classifying fiber %d of %d' % (i + 1, len(blob_data)))
        rc = r.coords
        im_class[rc[:,0],rc[:,1]] = c[i].astype('uint8')

    # Return useful stuff
    return im_class, blob_data

def load_classifier_from_file(classifier_file_string):
    # Loads in a classifier from a pick file

    import pickle
    svclassifier = pickle.load(open(classifier_file_string, 'rb'))

    return svclassifier


def implement_classifier(raw_image_file_string, classifier_file_string,
                         classifier_parameters=[],
                         image_to_label_parameters=[],
                         refine_fibers_parameters=[]):
    # Code uses a prior model to predict features

    from skimage.color import label2rgb
    from skimage.io import imsave

    # Turn raw_image_file_string into a labeled image
    if (classifier_parameters['verbose_mode']):
        print('Labeling image')
    im_label, im_sat, im_shuffled, im_gray = \
        im_proc.raw_image_file_to_labeled_image(
                        raw_image_file_string,
                        image_to_label_parameters=image_to_label_parameters)

    # Get the blob data
    if (classifier_parameters['verbose_mode']):
        print('Calculating blob properties')
    blob_data, region = im_proc.calculate_blob_properties(im_label)

    # Load the classifier
    if (classifier_parameters['verbose_mode']):
        print('Loading classifier')
    classifier_model = load_classifier_from_file(classifier_file_string)

    # Implement the classifier
    if (classifier_parameters['verbose_mode']):
        print('Classifying labeled image')
    im_class, blob_data = \
        classify_labeled_image(im_label, classifier_model)
        
    # Debug
    blob_data.to_excel("..\\temp\\classify_debug.xlsx")
    f_x, ax_x = plt.subplots(2,2, figsize=(10,10))
    ax_x[0,0].imshow(im_gray)
    ax_x[0,1].imshow(im_class)
    ax_x[1,0].imshow(im_label)
    im_s_x = im_proc.shuffle_labeled_image(im_label)
    ax_x[1,1].imshow(im_s_x)
    
#
#    # Deal with potentially connected fibers
#    if (classifier_parameters['verbose_mode']):
#        print('Handling potentially connected fibers')
#    im_class2, im_label2 = \
#        im_proc.handle_potentially_connected_fibers(im_class, im_label,
#                                                    blob_data, region,
#                                                    classifier_model,
#                                                    classifier_parameters['watershed_distance'],
#                                                    troubleshoot_mode=0)
#
##    # Shuffle im_label for improved display
##    im_shuffle = im_proc.shuffle_labeled_image(im_label)
##    im_shuffle2 = im_proc.shuffle_labeled_image(im_label2)
##
##    fig, ax = plt.subplots(3, 2, figsize=(10,10))
##    ax[0, 0].imshow(im_shuffle)
##    ax[0, 1].imshow(im_class)
##    ax[1, 0].imshow(im_label2)
##    ax[2, 0].imshow(im_shuffle2)
##    ax[2, 1].imshow(im_class2)
##
##    fig, ax = plt.subplots(3,2, figsize=(5,5))
##    for i in np.arange(1,4):
##        im_class_test = np.zeros(im_class2.shape)
##        im_class_test[im_class2==i] = 1
##        ax[(i-1),0].imshow(im_class_test)
#        
#    # Deduce the fiber seeds
#    im_fiber_seeds = np.zeros(im_class2.shape)
#    im_fiber_seeds[im_class2 == 1] = 1
#
#    im_refined = im_proc.refine_fiber_edges(im_fiber_seeds, im_sat,
#                                          refine_fibers_parameters = refine_fibers_parameters)
#    
#    
#    im_l2 = im_proc.label_image(im_refined)
#    r2 = im_proc.deduce_region_props(im_l2)
#    im_c2, bd2 = classify_labeled_image(im_l2, classifier_model)
#    
#    bd2.to_excel("..\\temp\\classify_test.xlsx")
#    
#    im_c3, im_l3 = \
#        im_proc.handle_potentially_connected_fibers(im_c2, im_l2,
#                                                    bd2, r2,
#                                                    classifier_model,
#                                                    classifier_parameters['watershed_distance'])
#
#    im_s3 = im_proc.shuffle_labeled_image(im_l3)
#
#    # Create overlay
#    im_mask = np.zeros(im_c3.shape)
#    im_mask[im_c3==1] = 1
#    im_overlay = label2rgb(im_mask, im_gray)
#    
#    fig,ax = plt.subplots(3,2, figsize=(12,7))
#    ax[0, 0].imshow(im_refined)
#    ax[0, 1].imshow(im_c2)
#    im_s2 = im_proc.shuffle_labeled_image(im_l2)
#    ax[1,0].imshow(im_s2)
#    ax[2,0].imshow(im_s3)
#    ax[2,1].imshow(im_c3)
#    
#
#    # Save final result
##    im_out = im_proc.merge_label_and_blue_image(im_mask, im_gray)
#    
#    im_out = im_proc.merge_rgb_planes(im_gray, np.zeros(im_gray.shape), im_mask)
#    
#    imsave(classifier_parameters['result_file_string'], im_out)
