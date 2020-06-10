from argparse import ArgumentParser, ArgumentTypeError
from itertools import combinations, product
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.stats import f
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd

# ==================================================================
# Helper functions
# ==================================================================
# ------------------------------
# Statistical functions
# ------------------------------

def rss(truths, estimates):
    subs = np.subtract(truths, estimates)
    subs = subs ** 2
    return np.sum(subs)

def adjusted_r2(r2, n, p):
    return 1-(1-r2)*(n-1)/(n-p-1)


def linear_regression_test(y_true, old_weights, old_x, new_weights, new_x):
    """
    F-test for multiple regression sequences
    :param y_true: True values of each sample
    :param old_weights: The weights of the old model
    :param old_x: The matrix of the old model
    :param new_weights: The weights of the new model
    :param new_x: The matrix of the new model
    :return: The p-value of the new model being distinct from the old
        p < threshold)
    """
    old_pred = old_weights * old_x
    old_pred = old_pred.apply(np.sum, axis=1)
    rss1 = rss(y_true, old_pred)
    new_pred = new_weights * new_x
    new_pred = new_pred.apply(np.sum, axis=1)
    rss2 = rss(y_true, new_pred)
    param_delta = len(new_weights)-len(old_weights)
    sample_delta = len(y_true)-len(new_weights)
    f_statistic = ((rss1-rss2)/param_delta)/(rss2/sample_delta)
    p = 1 - f.cdf(f_statistic, param_delta, sample_delta)
    
    return p

# ----------------------------    
# simplexify encoding function
# ----------------------------    
def simplex_encode(seq, wt_seq, combo):
    simplex_codes = []
    # sequence_length = len(wt_seq) # all input sequences should be the same length
    # Iterate through all combinations (nCr) of the genotype, where n=base_len and r=order+1.
    for c in combo:
        # for each position in each generated combination, assign a value of 1 if that position has a wt genotype and -1 if it has a different genotype
        position_values = [-1 if seq[i]==wt_seq[i] else 1 for i in c]
        # get the product of all position values
        simplex_value = np.prod(position_values)
        simplex_codes.append(simplex_value)
    return simplex_codes

# ------------------------------
# Dataframe manipulation
# ------------------------------    
# Helper function to mask-out columns not pertinent to the model
def exclude_columns(target_df, exclude=[]):
    return target_df.loc[:, ~target_df.columns.isin(exclude)]

def generate_label(positions):
    pos_str = '|'.join([str(x) for x in positions]) # editted by John, changed '-' to '|' to avoid autoconversion date formatting in excel without needing brackets
                                                    # technically not necessary, but most likely the users are using Excel rather than "text edit" or "notepad"
    return f"{pos_str}" # editted by John, removed outer brackets

# ------------------------------
# File path helper
# ------------------------------
def file_path(dir):
    path = Path(dir)
    if path.is_dir():
        raise ArgumentTypeError(
            "Provided path was a directory! Please provide a file path instead.")
    return path


# ==================================================================
# Epistasis analysis code
# ==================================================================
def main(data_input, 
        # default options no longer need to be supplied by arg parse
        genotype_col="Genotype", effect_col="Function", 
        position_output=Path.cwd().joinpath("position_out.csv"), 
        genotype_output=Path.cwd().joinpath("genotype_out.csv"), 
        r2_output=Path.cwd().joinpath("model_r2.csv")    ):
        
    # parse the input file and retrieve the WT sequence as well as the position labels
    base_seq = None
    position_labels = None
    skiprows = 4
    with open(data_input,'r') as f:
    
        first_line = next(f) # skip the first line, which is a header for 'WT'
        base_seq = next(f).strip().split(',')[0] # collect the second line, which is the WT sequence
            
        third_line = next(f) # skip the thord line, which is a header for "Position"
        position_labels = next(f).strip().split(',') # collect the fourth line, which contains the position numbers
            
    # Import our data
    df = pd.read_csv(data_input, usecols=[genotype_col, effect_col], skiprows=skiprows) 
    
    # genotypes = df.loc[:, genotype_col]
    effects = df.loc[:, effect_col]

    # Identify the changes (in preparation for simplex encoding)
    base_len = len(base_seq)
    deltas = [None for x in range(base_len)] # keep a list the same length as the wt sequence, but with the variant genotype at each position
    for gen in df[genotype_col]:
        for i in range(base_len):
            if base_seq[i]!=gen[i]:
                deltas[i] = gen[i]

    # Start fitting the models, iteratively increasing the order
    p_val = 0
    current_order = 0
    highest_order = 0
    excluded_columns = [genotype_col, effect_col] # for fitting of models, the genotype and effects are not needed as model parameters, this is used as input for the exclude_columns function
    # Generate a model with 0 parameters, always estimating the mean
    df['INTERCEPT'] = np.ones(df.shape[0], dtype='int8')
    # fit all data using just the intercepts column, which provides the model intercept
    best_model = LinearRegression(fit_intercept=False)
    best_model.fit(exclude_columns(df, excluded_columns), effects)
    
    # Track R2 and adjusted R2 for each increasingly complex model
    r2_values = [0] # set the first entry to zero for the intercept model
    adj_r2_values = [0]
    
    # store combinations for repeated use
    combo_by_order = {}
    
    # Fit increasingly complex models until an F-test fails
    while current_order < base_len:
    
        print("-------------------------------------") # print a line to the console to visually separate the output from each order
        
        # Initialize the new df values with the current order simplex labels
        combo = [c for c in combinations(range(base_len), current_order+1) ] # make all position combinations and store as a list
        combo_by_order[current_order] = combo # also store the combination so it can be retrieved by order later
        
        col_labels = ["Genotype"] + [generate_label([position_labels[i] for i in c]) for c in combo]
        addendum_df = pd.DataFrame(columns=col_labels)
        
        # Populate the new dataframe with the genotypes, in order to generate the simplex encoding
        addendum_df.Genotype = df.Genotype
        
        # each row is given the its own genotype, the wt sequence and the current set of position combinations for this effect order
        # in order to generate the simplex encoding of the genotype
        addendum_df = addendum_df.apply(lambda row: [row.Genotype] + simplex_encode(row.Genotype, base_seq, combo), axis=1, result_type='broadcast')
        
        # keep a record of the last order effect model, for testing against the current order to see if the current order 
        # offers a significant improvement over the last model
        last_model_encoding = exclude_columns(df, excluded_columns)  
        
        # join the new encoding with the previous encodings, essentially expanding the modelling parameters to the current order
        df = df.join(exclude_columns(addendum_df, ['Genotype']))
        
        # Isolate the modelling parameters from the genotype and effects
        current_model_encoding = exclude_columns(df, excluded_columns)
        # Fit the encodings of the current order to the observed effects
        new_model = LinearRegression(fit_intercept=False)
        new_model.fit(current_model_encoding, effects)

        # Assess the R2 and adjusted R2
        r2 = new_model.score(current_model_encoding, effects)
        r2_values.append(r2)
        adj_r2_values.append(adjusted_r2(r2, len(effects), len(new_model.coef_)))

        # Test via Likelihood Ratio Test if this new model fits better
        # Special case; skip 1st order, as linear_regression_test cannot test against it, because there is no other model to test against
        if (current_order > 0):
            param_count_2 = len(new_model.coef_) - 1  # Intercept doesn't count
            param_count_1 = len(best_model.coef_) - 1  # Intercept doesn't count
            print(f"No. Params: order{current_order}: {param_count_1} => order{current_order+1}: {param_count_2}")
            p_val = linear_regression_test(effects, best_model.coef_, last_model_encoding,
                        new_model.coef_, current_model_encoding)
            print(f"p value: {p_val}")

        # Prepare for the next cycle, if needed
        if p_val < 0.05: # the current model presents a significant improvement over the last model with fewer terms
            # save the current order model and proceed to the next order
            best_model = new_model
            highest_order = current_order
            current_order += 1
        else: # the model does not account for significantly more variance, stop the cycle
            break
            
    # let the user known the highest order model tested    
    print(f"Final Model Order: {highest_order+1}")

    # Determine the relative contributions of positional effects (R2)
    coef_r2s = []
    true_mean = np.mean(effects) # calculate the mean of the observed effects
    masked_df = exclude_columns(df, excluded_columns)
    
    # iterate through each column (representing a unique combination of interacting positions at a specific order) in the model encoding
    # and calculate the predicted effects for all genotypes. Then calculate the R2 of correlation between the predicted and observed effects.
    for i in range(len(best_model.coef_)):
        c = best_model.coef_[i]
        y_true = effects
        y_pred = masked_df.iloc[:, i]*c
        y_pred = y_pred+true_mean
        coef_r2s.append(r2_score(y_true, y_pred))
    
    # Reset the first value, which is the R2 of the intercept
    coef_r2s[0] = 0

    # Prepare to decode the coefficient's into genotypic effects
    current_order = 0 # restart again from the lowest order and climb up, purely for organization
    binary_code = [-1, 1] # encoding to let the program know whether a value should be labelled as WT (-1) or as the variant
    pos_iter = combo_by_order[current_order] 
    bin_iter = [[x] for x in binary_code ] 

    # Special case: the intercept is unique, being always positive
    effect_dict = {'INTERCEPT': best_model.coef_[0]}

    # Coefficient iterator index, start at 1 to skip the intercept, which is already included
    coef_i = 1
    # Assess remaining co-efficients, ignoring the intercept (see above)
    while coef_i < best_model.coef_.size:
        for pos in pos_iter:
            # Fetch the corresponding coefficient for this element
            c = best_model.coef_[coef_i]
            coef_i += 1
            for bin in bin_iter:
                # Generate the label for this element
                label = "|".join([
                    f"{base_seq[p]}.{position_labels[p]}" if b == -1
                    else f"{deltas[p]}.{position_labels[p]}"
                    for b, p in zip(bin, pos)
                ])

                # Calculate the effect
                effect = np.product(bin) * c
                effect_dict[label] = effect

        # Update the position and binary iterators, in preparation for next loop
        if current_order >= highest_order: # looped through all orders, stop 
            break
        else: # continue to next loop
            current_order += 1
            pos_iter = combo_by_order[current_order]
            bin_iter = list(product(binary_code, repeat=current_order+1))
            
    # Save our results
    with position_output.open('w') as pos_file:
        positional_df = pd.DataFrame(
            data=[exclude_columns(df, excluded_columns).columns, 2*best_model.coef_, coef_r2s],
            index=['indices', 'effect', 'R2'])
        positional_df = positional_df.T.set_index('indices')
        positional_df.to_csv(pos_file)

    with genotype_output.open('w') as gen_file:
        gentoptype_effects = pd.DataFrame.from_dict(effect_dict, orient='index')
        gentoptype_effects.to_csv(gen_file)

    with r2_output.open('w') as r2_file:
        delta_r2_values = [0]
        delta_r2_values[0] = r2_values[1]
        for i in range((len(r2_values)-2)):
            delta_r2=r2_values[i+2]-r2_values[i+1]
            delta_r2_values.append(delta_r2)
        indices = [f"order{n+1}" for n in range(-1, current_order+1)]
        indices = indices[1:]
        r2_values = r2_values[1:current_order+2]
        adj_r2_values = adj_r2_values[1:current_order+2]
        delta_r2_values = delta_r2_values[0:current_order+1]
        r2_df = pd.DataFrame({
            'model_order': indices,
            'R2': r2_values,
            '(Adjusted)': adj_r2_values,
            'Delta R2': delta_r2_values
        }).set_index('model_order')
        r2_df.to_csv(r2_file)

# ==================================================================
# Command line functionality
# ==================================================================
if __name__ == '__main__':
    # Parse the initial arguments
    parser = ArgumentParser(
        """
        Determine the most significant epistatic effects for a given 
        set of binary sequence changes.
        """
    )
    parser.add_argument('input_filepath', type=file_path,
                        help="""
                        The file containing the sequence and effect 
                        information, to be assessed for epistatic 
                        effects. Note that, per position, only two 
                        variations are allowed in the genotype sequence 
                        (treated as wild type and mut type), with the 
                        very first sequence found being treated as 
                        purely wild-type. All sequences should be of the 
                        same length.
                        """)

    args = parser.parse_args()
    main(args.input_filepath)
