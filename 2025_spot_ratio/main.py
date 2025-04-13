import scipy.stats as stats
import pandas as pd

def binomial_probability_table(n, p):
    """
    Generates a table showing cumulative probabilities in a binomial distribution.
    :param n: Total number of trials
    :param p: Probability of success in each trial
    :return: DataFrame containing cumulative probabilities with both percentage and scientific notation
    """
    # Compute binomial probabilities
    probabilities = {k: stats.binom.pmf(k, n, p) for k in range(n + 1)}

    # Compute cumulative probabilities for at least k occurrences
    cumulative_probabilities = {k: sum(probabilities[i] for i in range(k, n + 1)) for k in range(1, n + 1)}

    # Create a DataFrame
    df = pd.DataFrame.from_dict(cumulative_probabilities, orient='index', columns=['At Least Probability'])
    df.index.name = 'Number of Occurrences'

    # Format values to include both percentage and scientific notation
    df['At Least Probability'] = df['At Least Probability'].apply(lambda x: f"{x * 100:.9f}% ({x:.2e})")

    return df

# Example execution
def main(p):
    """
    Main function to generate and print binomial probability tables for multiple values of n.
    :param p: Probability of success
    """
    n_values = [1, 2, 3, 4, 5, 10, 15, 20]
    for n in n_values:
        print(f"\nResults for n={n}, p={p}")
        df = binomial_probability_table(n, p)
        print(df.to_string())


if __name__ == "__main__":
    # https://aws.amazon.com/ec2/spot/instance-advisor/
    p = 0.055 # 10min/3hours
    main(p)

