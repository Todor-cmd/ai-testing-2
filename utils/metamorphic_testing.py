import numpy as np



# Function to get predictions
def get_predictions(data, session):
    return session.run(None, {'X': data.values.astype(np.float32)})[0]

#  Audit Rate Invariance (Mean Age Test)
def mean_age_test(X_test, session):
    age_mean = X_test['persoon_leeftijd_bij_onderzoek'].mean()
    X_invariant = X_test.copy()
    X_invariant['persoon_leeftijd_bij_onderzoek'] = age_mean
    y_pred_original = get_predictions(X_test, session)
    y_pred_invariant = get_predictions(X_invariant, session)
    audit_rate_original = np.mean(y_pred_original)
    audit_rate_invariant = np.mean(y_pred_invariant)
    print(f"Audit Rate - Original: {audit_rate_original}, Invariant: {audit_rate_invariant}")
    print(f"Difference in Audit Rate: {audit_rate_original - audit_rate_invariant}")


#  Audit Rate Invariance (Mean Age Test)
def zero_age_test(X_test, session):
    X_invariant = X_test.copy()
    X_invariant['persoon_leeftijd_bij_onderzoek'] = 0
    y_pred_original = get_predictions(X_test, session)
    y_pred_invariant = get_predictions(X_invariant, session)
    audit_rate_original = np.mean(y_pred_original)
    audit_rate_invariant = np.mean(y_pred_invariant)
    print(f"Audit Rate (zero) - Original: {audit_rate_original}, Invariant: {audit_rate_invariant}")
    print(f"Difference in Audit Rate(zero): {audit_rate_original - audit_rate_invariant}")


# Sensitivity Test (Small Variations)
def sensitivity_test(X_test, session ,variations=[-5, +5]):
    y_pred_original = get_predictions(X_test, session=session)
    for variation in variations:
        X_sensitive = X_test.copy()
        X_sensitive['persoon_leeftijd_bij_onderzoek'] += variation
        y_pred_sensitive = get_predictions(X_sensitive, session)
        audit_rate_sensitive = np.mean(y_pred_sensitive)
        print(f"Audit Sensitivity - Variation {variation}: {audit_rate_sensitive}")

# Age Swapping Test
def age_swapping_test(X_test, session):
    y_pred_original = get_predictions(X_test, session)
    X_swap = X_test.copy()
    X_swap['persoon_leeftijd_bij_onderzoek'] = np.random.permutation(X_swap['persoon_leeftijd_bij_onderzoek'])
    y_pred_swapped = get_predictions(X_swap, session)
    swap_diff = np.mean(y_pred_original != y_pred_swapped)
    print(f"Age Swapping - Prediction Difference: {swap_diff}")


# Invariant Relation: Changing a Neighborhood Feature to a Constant
def invariant_relation_test(X_test, column_name, constant_value, session):
    # Mutate the specific neighborhood column to a constant value
    X_mutated = X_test.copy()
    X_mutated[column_name] = constant_value
    y_pred_original = get_predictions(X_test, session)
    y_pred_mutated = get_predictions(X_mutated, session)
    
    # Compare the predictions
    difference = np.mean(y_pred_original != y_pred_mutated)
    print(f"Invariant Test - Difference for {column_name}: {difference}")
    return difference

# Swap Relation: Swapping Neighborhood Features Between Individuals
def swap_relation_test(X_test, columns_to_swap, session):
    # Swap the values between the specified columns
    X_mutated = X_test.copy()
    X_mutated[columns_to_swap] = X_mutated[columns_to_swap].apply(np.random.permutation, axis=0)
    
    y_pred_original = get_predictions(X_test, session)
    y_pred_mutated = get_predictions(X_mutated, session)
    
    # Compare the predictions
    difference = np.mean(y_pred_original != y_pred_mutated)
    print(f"Swap Test - Difference after swapping {columns_to_swap}: {difference}")
    return difference


# do metamorphic test on region columns for model 1 because of the hint
def metamorphic_test_model_1(X_test, session):
    # List of neighborhood columns to test on
    neighborhood_columns = [
        'adres_recentste_wijk_charlois', 'adres_recentste_wijk_delfshaven', 'adres_recentste_wijk_feijenoord',
        'adres_recentste_wijk_ijsselmonde', 'adres_recentste_wijk_kralingen_c', 'adres_recentste_wijk_noord',
        'adres_recentste_wijk_other', 'adres_recentste_wijk_prins_alexa', 'adres_recentste_wijk_stadscentru',
        'adres_recentst_onderdeel_rdam', 'adres_recentste_buurt_groot_ijsselmonde', 'adres_recentste_buurt_nieuwe_westen',
        'adres_recentste_buurt_other', 'adres_recentste_buurt_oude_noorden','adres_recentste_buurt_vreewijk'
    ]
    
    # 1. Invariant Relation Test: Change a few neighborhood columns to a constant value and check the prediction difference
    print("\nRunning Invariant Relation Test:")
    for column in neighborhood_columns[:3]:  # Test on first 3 columns for invariant relation
        print(f"Testing column: {column}")
        invariant_relation_test(X_test, column, 0, session)  # Setting it to 0 for example

    # 2. Swap Relation Test: Swap all neighborhood columns and check the prediction difference
    print("\nRunning Swap Relation Test:")
    swap_relation_test(X_test, neighborhood_columns, session)  # Swap all columns



# do metamorphic tests on the age for model 2 because of the hint
def metamorphic_test_model_2(X, session):
    mean_age_test(X, session)
    zero_age_test(X, session)
    sensitivity_test(X, session)
    age_swapping_test(X, session)