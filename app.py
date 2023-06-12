import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from json import JSONEncoder
import streamlit as st

from Data_Generation.Dataset_Generation_Functions import make_boxes
from Data_Generation.Piecewise_Box_Functions import basic_box_array, forward_slash_array, combine_arrays, add_thickness
from Data_Plotting.Plot_TSNE import TSNE_reduction, plot_dimensionality_reduction
########################################################################################################################
# User Inputs
image_size = st.slider('Select a value for the image size', min_value=9, max_value=16)  # Max value is limited due to
# computational limitations of streamlit

density_selection = st.slider('Select a value for the number of equally spaced density values (0, 1]', min_value=1, max_value=10)
########################################################################################################################
# Compute Example Shapes

densities = np.linspace(0, 1, num=density_selection+1)[1:]

sample_basic_box = basic_box_array(image_size, 1)
sample_forward_slash_box = forward_slash_array(image_size, 1)
sample_combined = combine_arrays([sample_forward_slash_box, sample_basic_box])

sample_density = np.array([sample_combined * density_value for density_value in densities])
sample_thickness = []

for i in [1, 2, 3, 4]:
    copy = sample_combined
    test = add_thickness(copy, i)
    sample_thickness.append(test)

########################################################################################################################
# Output Example Shapes
st.write("Click 'Generate Samples' to show some density values that would exist in your dataset:")

# Show samples of various density values
if st.button('Generate Samples'):  # Generate the samples
    plt.figure(1)
    st.header("Sample Density Figures:")
    max_figures = min(density_selection, 5)  # Determine the number of figures to display
    for i in range(max_figures):
        plt.subplot(1, max_figures+1, i+1), plt.imshow(sample_density[i], cmap='gray', vmin=0, vmax=1)
        if i != 0:  # Show y-label for only first figure
            plt.tick_params(left=False, labelleft=False)
        plt.title("Density: " + str(round(densities[i], 4)), fontsize=6)
    plt.figure(1)
    # These settings can be used to display a colorbar
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax, shrink=0.1)
    st.pyplot(plt.figure(1))

    # Show samples of various thickness values
    st.header("Sample Thickness Figures:")
    plt.figure(2)
    for i in range(len(sample_thickness)):
        plt.subplot(1, 5, i+1), plt.imshow(sample_thickness[i], cmap='gray', vmin=0, vmax=1)
        if i != 0:  # Show y-label for only first figure
            plt.tick_params(left=False, labelleft=False)
        plt.title("Thickness: " + str(i+1), fontsize=6)
    plt.figure(2)
    # These settings can be used to display a colorbar
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax, shrink=0.1)
    st.pyplot(plt.figure(2))

########################################################################################################################
# Output Entire Dataset
st.write("Click 'Generate Dataset' to generate the dataset based on the conditions set previously:")
if st.button('Generate Dataset'):  # Generate the dataset
    boxes = make_boxes(image_size, densities)  # Create all the data points

    # Unpack all the data
    box_arrays, box_density, basic_box_thickness, forward_slash_box_thickness, back_slash_box_thickness,hot_dog_box_thickness, hamburger_box_thickness\
        = list(zip(*boxes))[0], list(zip(*boxes))[1], list(zip(*boxes))[2], list(zip(*boxes))[3], list(zip(*boxes))[4], list(zip(*boxes))[5], list(zip(*boxes))[6]

    # Plot TSNE of the data
    # Determine the labels of the TSNE Plot
    def flatten_array(array):  # define a function to flatten a 2D array
        return array.flatten()
    # apply the flatten_array function to each array in the list and create a list of flattened arrays
    flattened_arrays = np.array([flatten_array(a) for a in box_arrays])
    # calculate the average density for each array
    avg_density = np.sum(flattened_arrays, axis=1)/(np.shape(box_arrays[0])[0]*np.shape(box_arrays[0])[1])

    # Perform the TSNE Reduction
    x, y, title, embedding = TSNE_reduction(flattened_arrays)
    plt.figure(3)
    # set the color values for the plot
    plot_dimensionality_reduction(x, y, avg_density, title)
    # plt.title(title)
    plt.figure(3)
    st.pyplot(plt.figure(3))

    # Create a class to read the information from the generated CSV file
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    # Save the arrays in a JSON format so they can be read
    box_arrays = [json.dumps(x, cls=NumpyArrayEncoder) for x in box_arrays]

    # Create a dataframe to convert the data to a csv file
    dataframe = (pd.DataFrame((box_arrays, box_density, basic_box_thickness, forward_slash_box_thickness,
                               back_slash_box_thickness, hot_dog_box_thickness, hamburger_box_thickness)).T).astype(str)

    # Rename the columns to the desired outputs
    dataframe = dataframe.rename(
        columns={0: "Array", 1: "Density", 2: "Basic Box Thickness", 3: "Forward Slash Strut Thickness",
                 4: "Back Slash Strut Thickness", 5: "Vertical Strut Thickness", 6: "Horizontal Strut Thickness"})

    # Convert the dataframe to CSV
    csv = dataframe.to_csv()

    st.write("Here is what a portion of the generated data looks like (double click on the 'Array' cells to view the full array):")
    st.write(dataframe.iloc[:100,:])  # Display the data generated
    st.write("Click 'Download' to download a CSV file of the dataset:")
    st.download_button("Download Dataset", csv, file_name='2D_Lattice.csv')  # Provide download for user
