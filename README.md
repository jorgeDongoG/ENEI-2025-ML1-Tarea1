# Assignment: Linear Models, Regularization, and Model Selection on Real Data

**Deadline:** Sunday, October 5th, 2025, 23:59

**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

---

## Part A. Linear Regression From Scratch

1. **Dataset**
   Use the **California Housing dataset** (`from sklearn.datasets import fetch_california_housing`).

   * Create a hold-out test set.
   * Predict the median house value (`MedHouseVal`) from the remaining features.
   * Standardize features to zero mean and unit variance.

2. **Closed-form OLS**

   * Derive and implement $\hat\beta = (X^\top X)^{-1}X^\top y$ using only `numpy`.
   * Report coefficients and intercept.
   * Plot predicted vs. true median house value on a held-out test set.

3. **Gradient Descent**

   * Implement gradient descent to minimize mean squared error.
   * Experiment with at least two learning rates; show cost vs. iteration curves.
   * Compare parameters and test error to the closed-form OLS.

---

## Part B. Scikit-learn Linear Models

4. **Baseline**

   * Use `LinearRegression` and confirm the coefficients match your OLS implementation.
   * Compute $R^2$ and mean squared error on the test set.

---

## Part C. Regularization and Hyperparameter Choice

5. **Ridge and Lasso**

   * Fit `Ridge` and `Lasso` regressions for $\lambda$ values logarithmically spaced between $10^{-3}$ and $10^{2}$.
   * Plot coefficient magnitude vs. $\lambda$ (regularization paths).
   * Comment on which features shrink to (or toward) zero and why.

6. **k-Fold Cross-Validation**

   * Use `KFold` with 5 folds and `cross_val_score` to select the best $\alpha$ for both Ridge and Lasso.
   * Alternatively, demonstrate the convenience of `RidgeCV` and `LassoCV`.
   * Compare cross-validated test errors.

7. **Feature Engineering & Multicollinearity**

  * Add polynomial features (degree 2) using `PolynomialFeatures`.
  * Re-run Ridge/Lasso and discuss how regularization copes with the enlarged feature space.

---

## Part D. Bike Rentals

8. **Alternative Dataset**

  * Use the **Bike Sharing Dataset** (available in the `data` folder).
  * Predict daily rentals (`cnt`); investigate seasonal effects.
  * Apply all the same steps as above.

---

### Deliverables
You must fork the [original repository](), and turn in a link to your groups repository with:

* A Jupyter notebook (in the `src` folder) with:

  * Your `numpy` implementations for OLS and gradient descent,
  * Plots: cost-function convergence, coefficient paths, predicted vs. actual values.
* A write-up in Markdown. Replace the contents of this file (`README.md`) with:
  
  * The names of your group's members:
   - Jorge Alexis Dongo Gutierrez
   - Marcelo Sebastian Chavez Cisneros
   
### First data set:

  * Differences observed between OLS, Ridge, and Lasso,
The estimated coefficients and the errors obtained across all methods appear to be quite similar. However, while the execution time for OLS is negligible, Ridge and Lasso required more time with Lasso taking slightly longer. In a more complex scenario, we might see a tangible benefit from using Ridge or Lasso.

  * Effect of learning rate on gradient descent,
We experimented with learning rate coefficients of 0.4, 0.9, and 2. With 0.4 and 0.9, the algorithm converged to a solution, and the cost vs. iteration curve was steeper with the higher coefficient. Based on this, we decided to try an even larger coefficient to potentially achieve faster convergence. However, using a learning rate of 2 resulted in a process that failed to converge.

  * How k-fold cross-validation influenced the choice of regularization strength.

For this particular dataset, we did not observe any significant changes when applying k-fold cross-validation.

### Second data set:

  * Differences observed between OLS, Ridge, and Lasso,

Implementing OLS is computationally simpler, while Ridge and Lasso require iterating over a lambda. Regarding errors, the MSE of OLS, Ridge, and Lasso is the same; therefore, there is no difference in that metric for these three models. However, Lasso converges faster than Ridge, which takes more iterations to reach the optimum.

  * Effect of learning rate on gradient descent,

With a higher value, it converges faster; however, during the model calculation process, if we set a very high value, the iteration would fail to converge, and in my case, it caused my Jupyter kernel to crash.

  * How k-fold cross-validation influenced the choice of regularization strength.

In my case, k-fold did not substantially improve the model, as the mean squared error was practically the same. Where it did significantly improve the mean squared error was in the polynomial case, reducing it by almost 40%.
