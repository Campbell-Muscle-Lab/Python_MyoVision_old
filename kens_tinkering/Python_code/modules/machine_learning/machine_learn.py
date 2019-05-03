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


def implement_classifier(raw_image_file_string,
                         classifier_parameters=[],
                         image_to_label_parameters=[],
                         refine_fibers_parameters=[]):
    # Code uses a prior model to predict features
    
    from skimage.io import imread
    import gc

    # Save dictionary terms to reduce clutter
    verbose = image_to_label_parameters['verbose_mode']
    classifier_file_string = classifier_parameters['classifier_file_string']

    # Turn raw_image_file_string into a labeled image
    if (verbose):
        print('Labeling image')
    im_label, im_sat, im_gray = \
        im_proc.raw_image_file_to_labeled_image(
                        raw_image_file_string,
                        image_to_label_parameters=image_to_label_parameters)

    if (verbose):
        print('Calculating blob properties')
    blob_data, region = im_proc.calculate_blob_properties(im_label)

    # Load the classifier
    if (verbose):
        print('Loading classifier')
    classifier_model = load_classifier_from_file(classifier_file_string)

    # Implement the classifier
    if (verbose):
        print('Classifying labeled image')
    im_class, blob_data = \
        classify_labeled_image(im_label, classifier_model)

    # Deal with potentially connected fibers
    if (verbose):
        print('Handling potentially connected fibers')
    im_class2, im_label2 = \
        im_proc.handle_potentially_connected_fibers(im_class, im_label,
                                                    blob_data, region,
                                                    classifier_model,
                                                    classifier_parameters['watershed_distance'],
                                                    troubleshoot_mode=0)
    if (verbose):
        print('Deducing fiber seeds and refining edges')
    im_fiber_seeds = np.zeros(im_class2.shape)
    im_fiber_seeds[im_class2 == 1] = 1

    im_refined = im_proc.refine_fiber_edges(im_fiber_seeds, im_gray,
                                          refine_fibers_parameters = refine_fibers_parameters)

    if (verbose):
        print('Labeling image with refined edges')
    im_label3 = im_proc.label_image(im_refined)
    
    if (verbose):
        print('Calculating blob properties for image with refined edges')
    blob_data3, region3 = im_proc.calculate_blob_properties(im_label3)

    if (verbose):
        print('Re-classifying image with refined edges')
    im_class3, blob_data3 = classify_labeled_image(im_label3, classifier_model)

    if (verbose):
        print('Final pass to separate potentially connected fibers')
    im_final_classification, im_final_label = \
        im_proc.handle_potentially_connected_fibers(im_class3, im_label3,
                                                    blob_data3, region3,
                                                    classifier_model,
                                                    classifier_parameters['watershed_distance'])

    if (verbose):
        print('Creating final overlay')
    
    # Create a labeled image with a black background
    im_temp = im_proc.shuffle_labeled_image(im_final_label,
                                            bg_color=(0, 0, 0))
    # Overlay on original image with transparency
    im_b = im_proc.merge_rgb_planes(np.zeros(im_gray.shape),
                            np.zeros(im_gray.shape),
                            im_proc.normalize_gray_scale_image(im_gray))
    im_final_overlay = np.ubyte(0.5 * 255*im_temp + 0.5 * 255*im_b)
    del im_b
    del im_temp

    # Save classifier_steps file if required
    if (classifier_parameters['classification_steps_image_file_string']):
        if (verbose):
            print('Creating figure to show classification steps')

        # Need to load raw image here
        im = imread(raw_image_file_string)

        # Write image files, make and delete necessary images as we go
        # to save memory
        base_file_string = \
            classifier_parameters['classification_steps_image_file_string']
        create_image_file_for_classification_step(
                im, 'Original image', base_file_string,'original_image')
        create_image_file_for_classification_step(
                im_gray, 'Image as gray scale',
                base_file_string, 'gray_scaled')
        create_image_file_for_classification_step(
                im_sat, 'After saturation',
                base_file_string, 'saturated')
        
        im_shuffle = im_proc.shuffle_labeled_image(im_label)
        create_image_file_for_classification_step(
                im_shuffle, 'Initial segmentation',
                base_file_string, 'initial_segmentation')
        del im_shuffle
        
        create_image_file_for_classification_step(
                im_class, 'Initial classification',
                base_file_string, 'initial_classification')

        im_shuffle2 = im_proc.shuffle_labeled_image(im_label2)
        create_image_file_for_classification_step(
                im_shuffle2, 'Segmentation after initial separation of connected fibers',
                base_file_string, 'segmentation_after_initial_separation_of_connected_fibers')
        del im_shuffle2
        
        create_image_file_for_classification_step(
                im_class2, 'Classification after initial separation of connected fibers',
                base_file_string, 'classification_after_initial_separation_of_connected_fibers')
        create_image_file_for_classification_step(
                im_fiber_seeds, 'Fiber seeds',
                base_file_string, 'fiber_seeds')
        create_image_file_for_classification_step(
                im_refined, 'Refined fiber edges',
                base_file_string, 'refined_edges')

        im_shuffle3 = im_proc.shuffle_labeled_image(im_label3)
        create_image_file_for_classification_step(
                im_shuffle3, 'Segmentation after refined edges',
                base_file_string, 'segmentation_after_refined_edges')
        del im_shuffle3

        create_image_file_for_classification_step(
                im_class3, 'Classification after refined edges',
                base_file_string, 'classification_after_refined_edges')

        im_final_shuffle = im_proc.shuffle_labeled_image(im_final_label)
        create_image_file_for_classification_step(
                im_final_shuffle, 'Final segmentation',
                base_file_string, 'final_segmentation')
        del im_final_shuffle

        create_image_file_for_classification_step(
                im_final_classification, 'Final classification',
                base_file_string, 'final_classification')
        create_image_file_for_classification_step(
                im_final_overlay, 'Final overlay',
                base_file_string, 'final_overlay')

    return im_final_classification, im_final_label, im_final_overlay

def create_image_file_for_classification_step(im, im_title,
                                              base_file_string,
                                              file_label,
                                              verbose=1):
    # Creates a figure showing an image with a title and saves it to file

    # Insert file_label into the base_file_string
    output_file_string = base_file_string[:-4] + ('_%s' % file_label) + \
                            base_file_string[-4:]

    if (verbose):
        print('Writing %s' % output_file_string)

    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    ax.imshow(im)
    ax.set_title(im_title)

    plt.savefig(output_file_string, bbox_inches='tight')
    plt.close(fig=fig)