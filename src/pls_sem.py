"""
PLS-SEM (Partial Least Squares Structural Equation Modeling) utilities.
"""

import pandas as pd
import numpy as np
from semopy import Model
from semopy.stats import calc_stats


def run_pls_sem(data: pd.DataFrame, model_spec: str) -> dict:
    """
    Run a PLS-SEM model.

    Args:
        data: DataFrame containing the observed variables
        model_spec: Model specification string in semopy/lavaan syntax

    Returns:
        Dictionary containing model results
    """
    model = Model(model_spec)
    model.fit(data)

    results = {
        "estimates": model.inspect(),
        "stats": calc_stats(model),
    }

    return results


def print_results(results: dict) -> None:
    """Print formatted PLS-SEM results."""
    print("=" * 60)
    print("PLS-SEM Model Results")
    print("=" * 60)

    print("\nParameter Estimates:")
    print(results["estimates"].to_string())

    print("\nModel Statistics:")
    for stat, value in results["stats"].items():
        print(f"  {stat}: {value}")


# Example usage
if __name__ == "__main__":
    # Example model specification (lavaan syntax)
    example_model = """
    # Measurement model
    # LatentVar =~ indicator1 + indicator2 + indicator3

    # Structural model
    # DependentVar ~ IndependentVar1 + IndependentVar2
    """

    print("PLS-SEM module loaded successfully.")
    print("Use run_pls_sem(data, model_spec) to run your model.")
