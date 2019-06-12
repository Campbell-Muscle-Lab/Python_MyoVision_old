# Code for machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from modules.image_processing import image_proc as im_proc


def create_classifier_model(classification_parameters):
    # Code develops a classifier from manually annotated data

    from sklearn import svm
    import pickle

    # Load data
    print('Loading data for classification')
    d = pd.read_excel(classification_parameters['input_data_file_string'])

    # Parse data
    c = d['classification']
    d = d.drop(['classification', 'label'], axis=1)
    print(d.head())

    # Create classifier
    if (classification_parameters['classification_kernel'] == 'linear'):
        classifier = svm.SVC(kernel='linear')

    if (classification_parameters['classification_kernel'] == 'poly'):
        classifier = svm.SVC(kernel='poly',
                             order=classification_parameters['poly_order'])

    # Training classifier
    print('Training classifier')
    classifier.fit(d, c)

    # Save model
    classifier_output_file_string = \
        classification_parameters['output_classifier_file_string']

    # Make the directory if it does not exist
    dir_path = os.path.dirname(classifier_output_file_string)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    if (classifier_output_file_string):
        print('Saving classifier model to %s' % classifier_output_file_string)
        pickle.dump(classifier,
                    open(classifier_output_file_string, 'wb'))

#    # Make figure
#    print('Making classifier figure')
#    var_names = d.columns
#    no_of_variables = len(var_names)
#
#    # Make a figure
#    fig, ax = plt.subplots(no_of_variables, no_of_variables, figsize=(10, 10))
#    for i in np.arange(0, no_of_variables):
#        for j in np.arange(i, no_of_variables):
#            x = d.iloc[:, i]
#            y = d.iloc[:, j]
#
##            x_step = 0.01 * (x.max() - x.min())
##            y_step = 0.01 * (y.max() - y.min())
##
##            xm,ym = make_meshgrid(x,y,[x_step,y_step])
##            plot_contours(ax[i,j],classifier,xm,ym,
##                          cmap=plt.cm.coolwarm,s=20,edgecolors='k')
#
#            ax[i, j].scatter(x, y, c=c, alpha=0.2)
#            ax[i, j].set_xlabel(var_names[i])
#            ax[i, j].set_ylabel(var_names[j])
#
#    output_image_file_string=classification_parameters['output_image_file_string']
#    print('Saving figure to %s' % output_image_file_string)
#    plt.savefig(output_image_file_string)


def learn_test_2(input_data_file_string, output_classifier_file_string):

    from sklearn import svm

    # Load data
    d = pd.read_excel(input_data_file_string)

    c = d['classification']
    d = d.drop(['classification','label'], axis=1)
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
    X = d.drop(['label', 'classification'], axis=1)
    y = d['classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

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
    im_class = np.zeros(im_label.shape, dtype='uint8')
    for i, r in enumerate(region):
        if (np.mod(i + 1, 100) == 1):
            print('Classifying fiber %d of %d' % (i + 1, len(blob_data)))
        rc = r.coords
        im_class[rc[:, 0], rc[:, 1]] = c[i].astype('uint8')

    # Return useful stuff
    return im_class, blob_data, region


def load_classifier_from_file(classifier_file_string):
    # Loads in a classifier from a pick file

    import pickle
    svclassifier = pickle.load(open(classifier_file_string, 'rb'))

    return svclassifier


def implement_classifier(raw_image_file_string,
                         results_folder=[],
                         image_to_label_parameters=[],
                         classifier_parameters=[],
                         refine_fibers_parameters=[],
                         verbose=1):
    # Code uses a prior model to predict features

    from skimage.io import imread

    # Turn raw_image_file_string into a labeled image
    if (verbose):
        print('Labeling image')
    im_label, im_sat, im_gray = \
        im_proc.raw_image_file_to_labeled_image(
                        raw_image_file_string,
                        image_to_label_parameters=image_to_label_parameters,
                        results_folder=results_folder)

    # Load the classifier
    if (verbose):
        print('Loading classifier')
    classifier_model = load_classifier_from_file(
            classifier_parameters['classification_model_file_string'])

    # Implement the classifier
    if (verbose):
        print('Classifying labeled image')
    im_class, blob_data, region = \
        classify_labeled_image(im_label, classifier_model)

    # Deal with potentially connected fibers
    if (verbose):
        print('Handling potentially connected fibers')
    im_class2, im_label2 = \
        im_proc.handle_potentially_connected_fibers(im_class, im_label,
                                                    blob_data, region,
                                                    classifier_model,
                                                    image_to_label_parameters['watershed_distance'],
                                                    troubleshoot_mode=0)

    if (verbose):
        print('Deducing fiber seeds and refining edges')
    im_fiber_seeds = np.zeros(im_class2.shape)
    im_fiber_seeds[im_class2 == 1] = 1

    im_refined, im_gradient = im_proc.refine_fiber_edges(
                                im_fiber_seeds, im_gray,
                                refine_fibers_parameters = refine_fibers_parameters)

    if (verbose):
        print('Labeling image with refined edges')
    im_label3 = im_proc.label_image(im_refined)

    if (verbose):
        print('Re-classifying image with refined edges')
    im_class3, blob_data3, region3 = classify_labeled_image(im_label3,
                                                            classifier_model)

    if (verbose):
        print('Final pass to separate potentially connected fibers')
    im_final_classification, im_final_label = \
        im_proc.handle_potentially_connected_fibers(im_class3, im_label3,
                                                    blob_data3, region3,
                                                    classifier_model,
                                                    image_to_label_parameters['watershed_distance'],
                                                    troubleshoot_mode=1)
    im_final_classification = im_proc.fill_holes_in_non_binary_image(im_final_classification)

    im_final_label = im_proc.fill_holes_in_non_binary_image(im_final_label)

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

    # Deduce folder names for output of tracking steps
    if (results_folder):
        steps_base_file_string = os.path.join(results_folder,
                                              'processing',
                                              'classification_steps',
                                              'step.png')

        step_counter = 1
        create_image_file_for_classification_step(
                im_gray, 'Original image as gray_scale',
                steps_base_file_string, step_counter, 'original_gray')

        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_sat, 'After saturation',
                steps_base_file_string, step_counter, 'saturated')

        # This step is only used for visualization
        im_shuffle = im_proc.shuffle_labeled_image(im_label)
        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_shuffle, 'Initial segmentation',
                steps_base_file_string, step_counter, 'initial_segmentation')
        del im_shuffle

        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_class, 'Initial classification',
                steps_base_file_string, step_counter, 'initial_classification')

        # Delete images we no longer need to save memory
        del im_gray, im_sat, im_label, im_class

        # This step is only for visualization
        im_shuffle2 = im_proc.shuffle_labeled_image(im_label2)
        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_shuffle2, 'Segmentation after initial separation of connected fibers',
                steps_base_file_string, step_counter, 'segmentation_after_initial_separation_of_connected_fibers')
        del im_shuffle2

        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_class2, 'Classification after initial separation of connected fibers',
                steps_base_file_string, step_counter, 'classification_after_initial_separation_of_connected_fibers')
    
        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_fiber_seeds, 'Fiber seeds',
                steps_base_file_string, step_counter, 'fiber_seeds')

        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_gradient, 'Gradient image for refine fiber edges',
                steps_base_file_string, step_counter, 'gradient_image')

        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_refined, 'Refined fiber edges',
                steps_base_file_string, step_counter, 'refined_edges')

        del im_gradient, im_refined, im_label2, im_class2

        # This step just for visualization
        im_shuffle3 = im_proc.shuffle_labeled_image(im_label3)
        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_shuffle3, 'Segmentation after refined edges',
                steps_base_file_string, step_counter, 'segmentation_after_refined_edges')
        del im_shuffle3

        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_class3, 'Classification after refined edges',
                steps_base_file_string, step_counter, 'classification_after_refined_edges')

        # This step just for visualization
        im_final_shuffle = im_proc.shuffle_labeled_image(im_final_label)
        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_final_shuffle, 'Final segmentation',
                steps_base_file_string, step_counter, 'final_segmentation')
        del im_final_shuffle

        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_final_classification, 'Final classification',
                steps_base_file_string, step_counter, 'final_classification')
        
        step_counter = step_counter + 1
        create_image_file_for_classification_step(
                im_final_overlay, 'Final overlay',
                steps_base_file_string, step_counter, 'final_overlay')

    return im_final_classification, im_final_label, im_final_overlay


def create_image_file_for_classification_step(im, im_title,
                                              base_file_string,
                                              step_counter,
                                              file_label,
                                              verbose=1):
    # Creates a figure showing an image with a title and saves it to file
    import os

    # Insert file_label into the base_file_string
    output_file_string = base_file_string[:-4] + \
                            ('_%d_%s' % (step_counter, file_label)) + \
                            base_file_string[-4:]

    if (verbose):
        print('Writing %s' % output_file_string)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(im)
    ax.set_title(im_title)

    # Check directory exists and make it if necessary
    dir_path = os.path.dirname(output_file_string)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # Save the figure
    plt.savefig(output_file_string, bbox_inches='tight')
    plt.close(fig=fig)
