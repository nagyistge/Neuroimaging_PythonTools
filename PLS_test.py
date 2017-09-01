def run_PLSR(x_original, y_original):
    import numpy as np
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import scale
    from sklearn import cross_validation

    # Model tuning: selecting the number of components
    X_train, X_test , y_train, y_test = cross_validation.train_test_split(x_original, y_original, test_size=0.5, random_state=42)
    errors = list()
    kf_10 = cross_validation.KFold(len(X_test), n_folds=10, shuffle=True, random_state=1)

    for number_of_components in np.arange(1,x_original.shape[1]):
        pls = PLSRegression(n_components=number_of_components)
        fit = pls.fit(scale(X_train), y_train)
        mse = cross_validation.cross_val_score(pls, scale(X_train), y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
        errors.append(-mse)

    number_of_components = np.arange(1,11)[np.where(errors == np.min(errors))[0][0]]

    # Model evaluation: testing the best model
    pls = PLSRegression(n_components=number_of_components)
    pls.fit(X_train, y_train)
    RMSE = -cross_validation.cross_val_score(pls, scale(X_test), y_test, cv=kf_10, scoring='neg_mean_squared_error').mean()
    accuracy = accuracy_score(y_test, pls.predict(X_test).round(), normalize=True)

    # Comparison to shuffled data
    # Calculating the cross-validated score for the original sample
    kf_10 = cross_validation.KFold(len(X_test), n_folds=10, shuffle=True, random_state=1)
    pls = PLSRegression(n_components=number_of_components)
    pls.fit(X_train, y_train)
    original_score = -cross_validation.cross_val_score(pls, scale(X_test), y_test, cv=kf_10, scoring='neg_mean_squared_error').mean()

    # Comparison to a random sample
    from sklearn.model_selection import permutation_test_score
    _,_,permutation_p = permutation_test_score(pls, X_test, y_test)

    print('Number of components: ' + str(number_of_components))
    print('RMSE: %.3f' % RMSE)
    print('Prediction accuracy: %.3f' % accuracy)
    print('Test compared to permuted sample: p=%.3f' % permutation_p)

    return pls


# Testing the implementation with a validation sample
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# This dataset has a categorical outcome, i.e. species of the sample
iris = datasets.load_iris()
x_model, x_validation, y_model, y_validation = train_test_split(iris.data, iris.target, test_size=0.2)
pls = run_PLSR(x_model, y_model)
validation_score = accuracy_score(y_validation, pls.predict(x_validation).round(), normalize=True)
print('Validation accuracy: %.2f' % validation_score)
