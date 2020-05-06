from argparse import ArgumentParser, ArgumentTypeError
from itertools import combinations, product
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd


def generate_label(positions):
    pos_str = '-'.join([str(x) for x in positions])
    return f"({pos_str})"


def rss(truths, estimates):
    subs = np.subtract(truths, estimates)
    subs = subs ** 2
    return np.sum(subs)


def adjusted_r2(r2, n, p):
    return 1-(1-r2)*(n-1)/(n-p-1)


def lrt(y_true, old_weights, old_x, new_weights, new_x):
    """
    Likelihood ratio test for multiple regression sequences
    :param y_true: True values of each sample
    :param old_weights: The weights of the old model
    :param old_x: The matrix of the old model
    :param new_weights: The weights of the new model
    :param new_x: The matrix of the new model
    :return: The p-value of the new model being distinct from the old
        p > threshold, unlike CDF (p < threshold)
    """
    # Helper function to save some redundant code later
    def get_log_likelihood(truths, weights, matrix):
        n = len(truths)
        preds = weights * matrix
        preds = preds.apply(np.sum, axis=1)
        sum_comp = np.sum((truths - preds) ** 2)
        s2 = np.var(preds)
        sum_comp = sum_comp / (2 * s2)
        log_like = -(n / 2) * np.log(2 * np.pi) - n * np.log(s2) - sum_comp
        return log_like

    # Old Likelihood calculation
    old_log_like = get_log_likelihood(y_true, old_weights, old_x)

    # New Likelihood calculation
    new_log_like = get_log_likelihood(y_true, new_weights, new_x)

    like_ratio = -2*(new_log_like - old_log_like)

    p = chi2.sf(like_ratio, len(new_weights)-len(old_weights))

    return p


def main(data_input, genotype_col, effect_col, position_output,
         position_labels, genotype_output, plot_output, r2_output):
    # Import our data
    df = pd.read_csv(data_input, usecols=[genotype_col, effect_col])
    genotypes = df.loc[:, genotype_col]
    effects = df.loc[:, effect_col]

    # Identify the changes (in preparation for simplex encoding)
    base_seq = genotypes.iloc[0]
    base_len = len(base_seq)
    deltas = [None for x in range(base_len)]
    for gen in genotypes:
        diffs = [i for i in range(base_len) if base_seq[i] != gen[i]]
        for i in diffs:
            if deltas[i] is None:
                deltas[i] = gen[i]

    # Clean up our fields in preparation
    if not position_labels:
        position_labels = range(base_len)

    # Start fitting the models, iteratively increasing the order
    p_val = 1
    current_order = 0
    # Generate a model with 0 parameters, always estimating the mean
    df['INTERCEPT'] = np.ones(df.shape[0], dtype='int8')

    # Helper function to mask-out columns not pertinent to the model
    def mask_df(target_df):
        return target_df.loc[:, ~target_df.columns.isin([genotype_col, effect_col])]

    best_model = LinearRegression(fit_intercept=False)
    best_model.fit(mask_df(df), effects)
    # Track R2 and adjusted R2 for each increasingly complex model
    r2_values = [0]  # Guessing the mean is
    adj_r2_values = [0]
    # Fit increasingly complex models until an F-test fails
    while p_val > 0.05 and current_order < len(base_seq):
        # Initialize the new df values with the current order simplex labels
        pos_combos = combinations(position_labels, current_order + 1)
        col_labels = ["Genotype"]
        col_labels.extend([generate_label(p) for p in pos_combos])
        addendum_df = pd.DataFrame(columns=col_labels)

        # Generate the simplex encoded versions of each genotype
        addendum_df.Genotype = df.Genotype

        # Helper function to allow "apply" usage with data frames
        def simplexify(row):
            new_row = [row.Genotype]
            for c in combinations(range(base_len), current_order + 1):
                pos_gens = [(i, row.Genotype[i]) for i in c]
                pos_simple_vals = [-1 if base_seq[i] == g else 1 for i, g in pos_gens]
                simple_val = np.prod(pos_simple_vals, dtype='int8')
                new_row.append(simple_val)
            return new_row

        addendum_df = addendum_df.apply(simplexify, axis=1, result_type='broadcast')

        # Join the new data to our existing data
        del addendum_df['Genotype']
        old_df = df  # Cached for later; saves a filtering operation
        df = df.join(addendum_df)

        # Fit the model to this data
        new_model = LinearRegression(fit_intercept=False)
        new_model.fit(mask_df(df), effects)

        # Assess the R2 and adjusted R2
        r2 = new_model.score(mask_df(df), effects)
        r2_values.append(r2)
        adj_r2_values.append(adjusted_r2(r2, len(effects), len(new_model.coef_)))

        # Test via Likelihood Ratio Test if this new model fits better
        # Special case; skip 0th order, as LRT cannot test against it
        if (current_order > 0):
            param_count_2 = len(new_model.coef_) - 1  # Intercept doesn't count
            param_count_1 = len(best_model.coef_) - 1  # Intercept doesn't count
            print(f"No. Params: {param_count_1} => {param_count_2}")
            p_val = lrt(effects, best_model.coef_, mask_df(old_df),
                        new_model.coef_, mask_df(df))
            print(f"p value: {p_val}")

        # Prepare for the next cycle, if needed
        if p_val > 0.05:
            best_model = new_model
            current_order += 1
        print("-------------------------------------")
    print(f"Final Model Order: {current_order - 1}")

    # Determine the relative contributions of positional effects (R2)
    coef_r2s = []
    true_mean = np.mean(effects)
    masked_df = mask_df(df)
    for i in range(len(best_model.coef_)):
        c = best_model.coef_[i]
        y_true = effects
        y_pred = masked_df.iloc[:, i]*c
        y_pred = y_pred+true_mean
        coef_r2s.append(r2_score(y_true, y_pred))

    # Reset the first value, as it is not relevant
    coef_r2s[0] = 0

    # Prepare to decode the coefficient's into genotypic effects
    current_order = 1
    binary_code = [-1, 1]
    pos_iter = [[i] for i in range(base_len)]
    bin_iter = list(combinations(binary_code, 1))

    # Special case: the intercept is unique, being always positive
    effect_dict = {'INTERCEPT': best_model.coef_[0]}

    # Coefficient iterator index, tracked for later
    coef_i = 1

    # Assess remaining co-efficients, ignoring the intercept (see above)
    while coef_i < best_model.coef_.size:
        for pos in pos_iter:
            # Fetch the corresponding coefficient for this element
            c = best_model.coef_[coef_i]
            coef_i += 1
            for bin in bin_iter:
                # Generate the label for this element
                label = "".join([
                    f"{base_seq[p]}{position_labels[p]}" if b == -1
                    else f"{deltas[p]}{position_labels[p]}"
                    for b, p in zip(bin, pos)
                ])

                # Calculate the effect
                effect = np.product(bin) * c
                effect_dict[label] = effect

        # Update the position and binary iterators, in preparation for next loop
        current_order += 1
        pos_iter = combinations(range(base_len), current_order)
        # Leave the list; it breaks without it, and my sleep addled
        # brain cannot understand why
        bin_iter = list(product(binary_code, repeat=current_order))

    # Save our results
    with position_output.open('w') as pos_file:
        positional_df = pd.DataFrame(
            data=[mask_df(df).columns, best_model.coef_, coef_r2s],
            index=['indices', 'effect', 'R2'])
        positional_df = positional_df.T.set_index('indices')

        positional_df.to_csv(pos_file)

    with genotype_output.open('w') as gen_file:
        gentoptype_effects = pd.DataFrame.from_dict(effect_dict, orient='index')
        gentoptype_effects.to_csv(gen_file)

    with r2_output.open('w') as r2_file:
        indices = [f"order{n}" for n in range(-1, current_order-1)]
        indices[0] = 'mean'
        r2_df = pd.DataFrame({
            'model_order': indices,
            'R2': r2_values,
            '(Adjusted)': adj_r2_values
        }).set_index('model_order')
        r2_df.to_csv(r2_file)

    # Save a plot of the data, if requested
    if plot_output:
        # Initial setup
        plt.figure(dpi=200, figsize=(19.2, 10.8))
        plt.plot(mask_df(df), best_model.coef_)
        plt.xticks(rotation=90)
        # Drawing order-distinction lines
        current_len = 1
        line_idx = [1]
        for i in range(df.shape[1] - 2):
            pos = i + 2
            l = len(df.columns[pos].split(","))
            if current_len != l:
                current_len = l
                line_idx.append(i)
        for idx in line_idx:
            plt.axvline(idx, ls=":", c="red")

        # Draw the center-line to more easily distinguish small effects
        plt.axhline(0, ls=":", c="black")

        # Axis labels
        plt.xlabel("position(s)")
        plt.ylabel("effect")

        plt.savefig(plot_output)


def file_path(dir):
    path = Path(dir)
    if path.is_dir():
        raise ArgumentTypeError(
            "Provided path was a directory! Please provide a file path instead.")
    return path


if __name__ == '__main__':
    # Parse the initial arguments
    parser = ArgumentParser(
        """
        Determine the most significant epistatic effects for a given 
        set of binary sequence changes.
        """
    )
    parser.add_argument('-i', '--data_input', type=file_path,
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
    parser.add_argument('--genotype_col', default='Genotype',
                        help="""
                        The name of the column to use when determining 
                        the genotype of a row. Defaults to "Genotype" if 
                        not specified.
                        """)
    parser.add_argument('--effect_col', default='Function',
                        help="""
                        The name of the column to use when determining 
                        the corresponding effect for the genotype (for 
                        example, reaction rate, temperature etc.). 
                        Defaults to "Function" if not specified
                        """)
    parser.add_argument('--position_labels', default=None, nargs='*',
                        help="""
                        The label IDs to assign to each position. If not 
                        specified, defaults to 0 indexed integers up to
                        the length of the sequence (0, 1, 2 etc.). If 
                        specified, the number of labels specified must 
                        match the length of the sequences being tested.
                        """)
    parser.add_argument('--position_output',
                        default=Path.cwd().joinpath("position_out.csv"),
                        type=file_path,
                        help="""
                        The file the estimated positional effects should 
                        be saved in. Defaults to the directory this 
                        script is being run in, named "position_out.csv".
                        """)
    parser.add_argument('--genotype_output',
                        default=Path.cwd().joinpath("genotype_out.csv"),
                        type=file_path,
                        help="""
                        The file the estimated genotype interaction 
                        effects should be saved in. Defaults to the 
                        directory this script is being run in, named
                        "genotype_out.csv".
                        """)
    parser.add_argument('--r2_output',
                        default=Path.cwd().joinpath("model_r2.csv"),
                        type=file_path,
                        help="""
                        The file the estimated genotype interaction 
                        effects should be saved in. Defaults to the 
                        directory this script is being run in, named
                        "model_r2.csv".
                        """)
    parser.add_argument('--plot_output',
                        default=None,
                        type=file_path,
                        help="""
                        The file the positional value plot should be
                        saved to. If not specified, no plot is generated.
                        """)

    args = parser.parse_args()
    main(**vars(args))
