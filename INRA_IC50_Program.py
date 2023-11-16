#!/usr/bin/env python
import os
variable_value = os.environ.get("PATH")
'''
-------------------------------------------------------------------------------
Title: INRAE_IC50_ANALYSIS
Group: Microbial Tecnologies, CSIRO
Author: Ned Walsh

Synopsis:

-------------------------------------------------------------------------------


Importing Libraries
SciPy:
Numpy:
os and sys:
Pandas:

'''

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import numpy as np
import glob
import sys
import pandas as pd
'''
-------------------------------------------------------------------------------
Defining Functions

'''


def is_numeric(s):
    try:
        # Try to convert the string to an integer
        int(s)
        return True
    except ValueError:
        try:
            # Try to convert the string to a float
            float(s)
            return True
        except ValueError:
            return False

def nonlinear_model(x, a, b,c):
    return a * np.log2(x) + b

def sigmoid_function(x, a, b, c, d):
    return (c / (1 + np.exp(-a * (x - b)))) + d

def remove_outliers_zscore(x_data, y_data, threshold=2):
    z_scores = np.abs((y_data - np.mean(y_data)) / np.std(y_data))
    valid_indices = z_scores < threshold
    return x_data[valid_indices], y_data[valid_indices]

def polynomial_line(x, *coeffs):
    return np.polyval(coeffs, x)

def tanh_function(x, a, b, c, d, e):
    return a * np.tanh((b * x) + c) + d

def linear_model_line(x, a, b):
    return a * x + b

def equation_to_solve(x, coeffs, a, b):
    return polynomial_line(x, coeffs) - linear_model_line(x, a, b)

def stack_split_coords(array1, array2):
    result = np.dstack((array1, array2))
    result_list = result.tolist()
    x_coords = []
    y_coords = []
    for i in result_list:
        for x in i:
            x_coords.append(x[0])
            y_coords.append(x[1])
    return x_coords, y_coords

def NaN_control(x_coords, y_coords):
    x_final = []
    y_final = []
    for it1, it2 in zip(x_coords, y_coords):
        if not np.isnan(it1) and not np.isnan(it2):
            x_final.append(it1)
            y_final.append(it2)
    return x_final, y_final

def coord_slices(slices, concentrations, abs_results):
    coord_dict = {}
    for num, i in enumerate(slices):
        start, end = map(int, i.split(':'))
        name = f'{num}_{concentrations[1,start].split(" ")[0]}'
        dil = float(concentrations[3, start])
        print(dil)
        array1 = np.log(concentrations[4:,start:end].astype(float))/np.log(dil)
        array2=abs_results[1:,start:end]
        conc = concentrations[2,start]
        dil_fac = float(concentrations[3,start])
        coords = []
        length = len(array1[1,:])
        for numero in range(length):
            aliquot1 = array1[:,numero]
            aliquot2 = array2[:,numero]
            x_coords, y_coords = stack_split_coords(aliquot1, aliquot2)
            x_final, y_final = NaN_control(x_coords, y_coords)
            coords.append([x_final, y_final])
        coord_dict[name] = [conc, dil_fac, coords]
    return coord_dict

def control_slice(ctrls, test_data):
    output = {}
    for num, i in enumerate(test_data):
        ctrl = ctrls[num]
        if ctrl not in output:
            output[ctrl] = [i]
        else:
            output[ctrl].append(i)
    return output

def absorbance_open_tranform(data):
    botty = pd.read_excel(data)
    raw1 = botty[12:22].values
    raw2 = botty[23:].values

    for i in [raw1, raw2]:
        val = i[0][1]
        #print(val)
        if "570" in val:
            Raw570 = i[2:,1:].astype(float)
        elif "600" in val:
            Raw600 = i[2:,1:].astype(float)

    Raw570600 = Raw600 / Raw570 
    w_col_nums = np.column_stack((raw1[2:,0], Raw570600))
    final_data = np.vstack((raw1[1,:], w_col_nums))
    return final_data

def plot_sigmoid(x_data_cleaned, y_data_cleaned, pos, Name, conc, color):
    initial_guess = [1, np.min(x_data_cleaned)+((np.max(x_data_cleaned) - np.min(x_data_cleaned))/2),1.3, 0.1]
    # Example: bounds = ([a_min, b_min, c_min, d_min], [a_max, b_max, c_max, d_max])
    bounds = ([0.2, 0, 0.8, 0], [np.inf, np.inf, 1.4, 0.2])
    params, covariance = curve_fit(sigmoid_function, x_data_cleaned, y_data_cleaned, p0=initial_guess, bounds = bounds)
    a_fit, b_fit, c_fit, d_fit = params
    IC50 = 1.5**b_fit
    print(f'params are {params}, b_fit is {b_fit} IC50 is {IC50}')
    x_fit = np.linspace(min(x_data_cleaned), max(x_data_cleaned), 1000)  # Create a range of x values for the fitted curve
    a_fit, b_fit, c_fit, d_fit = params
    y_fit = sigmoid_function(x_fit, a_fit, b_fit, c_fit, d_fit)
    pos.plot(x_fit, y_fit, color=color, linewidth=2, label='Sigmoid Curve')
    pos.scatter(b_fit, (c_fit / 2 + d_fit), marker='o', color='green', label='Fitted Parameters', s=50)
    pos.annotate(f'IC50={IC50:.2f}{conc}', (b_fit, c_fit / 2 + d_fit), textcoords="offset points", xytext=(0,15), fontsize = 14, fontweight = "bold",
        color='red', ha='right', va='bottom')
    pos.hlines(y=(c_fit / 2 + d_fit), xmin=np.min(x_data_cleaned), xmax=np.max(x_data_cleaned), colors='black', linestyles='--')
    return IC50

"""
Main Graphing Function
"""


def ic_50_output_15dil(filename, data_dict, output_dir, controls):

    num_rows = 3
    num_cols = 1

    pltwid, plthgt = 16 , num_rows*4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(pltwid, plthgt), facecolor="white")

    IC50_dict = {} 

    keys = [i for i in data_dict]

    myC = []
    for num, Name in enumerate(keys):
        
        data = data_dict[Name]

        conc = data [0]
        dil_fac = data [1]

        for color, rep in enumerate(data[2]):
            if rep != [[], []]:
                colours = ['blue', 'green', 'orange', 'black']

                colour = colours[color]

                x_data = np.array(rep[0]).astype(float)
                y_data = np.array(rep[1]).astype(float)

                ax_row = num % num_rows  # Calculate row index in the subplot grid

                sorted_indices = np.argsort(x_data)

                x_data_cleaned = x_data[sorted_indices.astype(int)]
                y_data_cleaned = y_data[sorted_indices.astype(int)]
                print(x_data_cleaned)
                #x_data_cleaned, y_data_cleaned = remove_outliers_zscore(x_sorted, y_sorted)

                axes[ax_row].set_ylim([0, 1.5])

                if "hyg" in Name:
                    axes[ax_row].set_xlim([np.min(x_data_cleaned), np.max(y_data_cleaned)]) 

                axes[ax_row].scatter(x_data_cleaned, y_data_cleaned, color=colour)
                
                #PLOT THE SIGMOID CURVE AGAINST THE DATA

                y_data_range = np.max(y_data_cleaned)-np.min(y_data_cleaned)
                y_data_mean = np.mean(y_data_cleaned)
                y_data_std = np.std(y_data_cleaned)

                regression_coeffs = np.polyfit(x_data_cleaned, y_data_cleaned, 1)
                regression_line = np.polyval(regression_coeffs, x_data_cleaned)

                slope, intercept = regression_coeffs
                angle_rad = np.arctan2(slope, 1)
                angle_deg = np.degrees(angle_rad)
                print(angle_deg)

                if angle_deg < 2:
                    axes[ax_row].plot(x_data_cleaned, regression_line, color='red', linewidth=2, label='Regression Line')
                    if y_data_mean > 0.8:
                        myIC50 = "<3"
                        pass
                    else:
                        myIC50 = '>50'
                        pass
                
                elif 2<angle_deg<5:
                    #axes[ax_row].plot(x_data_cleaned, regression_line, color='red', linewidth=2, label='Regression Line')
                    #MAKE SUBSET OF THE DATA FOR SIGMOIDAL ANALYSIS
                    cut = len(x_data_cleaned) //2
                    x_data_subset = x_data_cleaned[:cut]
                    y_data_subset = y_data_cleaned[:cut]
                    try:
                        myIC50 = plot_sigmoid(x_data_cleaned, y_data_cleaned,axes[ax_row], Name, conc, colour)
                    except RuntimeError:
                        print(f'Could not fit Sigmoid Curve to {Name}')
                        myIC50 = 'NaN'
                        pass
                    except ValueError:
                        print(f'Could not fit Sigmoid Curve to {Name}')
                        myIC50 = 'NaN'
                        pass
                else:
                    try:
                        myIC50 = plot_sigmoid(x_data_cleaned, y_data_cleaned,axes[ax_row], Name, conc, colour)
                    except RuntimeError:
                        print(f'Could not fit Sigmoid Curve to {Name}')
                        myIC50 = 'NaN'
                        pass
                    except ValueError:
                        print(f'Could not fit Sigmoid Curve to {Name}')
                        myIC50 = 'NaN'
                        pass
                myC.append(str(myIC50))
                
        # Add labels and a title
        axes[ax_row].set_xlabel('Concentration log2(uM)')
        axes[ax_row].set_ylabel('Inhibition')
        axes[ax_row].set_title(Name, color='black', fontweight = "bold", size = 14)
        #axes[ax_row].set_ylim(0, 1.5) 

    plt.tight_layout()  # Adjust spacing between subplots
    
    output_directory = f"{output_dir}\\IC50_Graphs"

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    num_files = len(glob.glob(f"{output_directory}\\*.JPG"))
    plt.savefig(f'{output_directory}\\{filename}_{num_files}.JPG')
    plt.close()
    
    myli = []
    for key, val in controls.items():
        myli.append(str(np.mean(val)))
    Inhib = '\t'.join(myC)
    ctrls = '\t'.join(myli)

    outstr = f'{filename}\t{ctrls}\t{Inhib}\n'
    with open(f'{output_dir}\\INRAE_ANALYSIS.tsv', 'a') as outfile:
        outfile.write(outstr)

    return

"""
------------------------------------------------------------------------------
Data Input and Function Calling
------------------------------------------------------------------------------
"""

if __name__ == "__main__":
    input_dir=sys.argv[1]
    concentration_name = sys.argv[2]
    absorbance_name = sys.argv[3]

    os.chdir(input_dir)

    concent = pd.read_csv(concentration_name, header=None)
    concentrations = concent.values
    absorbance_results = pd.read_csv(absorbance_name, header=None)
    abs_results = absorbance_results.values

    slices = ['1:5','5:8','8:12']
    my_slices = coord_slices(slices, concentrations, abs_results)
    print(my_slices)
    ic_50_output_15dil("OUTPUT.jpg", my_slices)