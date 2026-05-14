# Decision Trees & Ensembles: Complete Reference Guide
**Shourya Choudhary**
**ARIES IIT Delhi | ML Lecture Series**

This document provides the mathematical foundations, implementation details, and advanced topics for decision trees and ensemble methods. Use this as a comprehensive reference after the lecture.

---

## Table of Contents

1. [Mathematical Foundations](#mathematical-foundations)
2. [Decision Tree Algorithm Details](#decision-tree-algorithm-details)
3. [Pruning Strategies](#pruning-strategies)
4. [Random Forest Deep Dive](#random-forest-deep-dive)
5. [Gradient Boosting Internals](#gradient-boosting-internals)
6. [Hyperparameter Tuning Guide](#hyperparameter-tuning-guide)
7. [Advanced Topics](#advanced-topics)
8. [Coding Assignment](#coding-assignment)
9. [Additional Resources](#additional-resources)

---

## Mathematical Foundations

### 1. Classification: Impurity Measures

#### Gini Impurity

**Definition**: Measures the probability of incorrectly classifying a randomly chosen element if it were randomly labeled according to the distribution of labels in the subset.

**Formula**:
```
Gini(S) = 1 - Σ(p_i)²
```

where `p_i` is the proportion of class `i` in set `S`.

**Derivation**:
- Probability of picking an element from class `i`: `p_i`
- Probability of misclassifying it (labeling it as any other class): `1 - p_i`
- Expected misclassification rate: `Σ p_i(1 - p_i) = Σ p_i - Σ(p_i)² = 1 - Σ(p_i)²`

**Properties**:
- Range: [0, 0.5] for binary classification
- Gini = 0: Pure node (all samples are the same class)
- Gini = 0.5: Maximum impurity (equal distribution in binary case)
- Gini for multi-class: max impurity = (K-1)/K where K = number of classes

**Example**:
```
Node with 7 Yes, 3 No:
Gini = 1 - (0.7)² - (0.3)² = 1 - 0.49 - 0.09 = 0.42

Pure node with 10 Yes, 0 No:
Gini = 1 - (1.0)² - (0.0)² = 0.0
```

#### Entropy / Information Gain

**Definition**: Measures the average amount of information needed to identify the class of an element.

**Formula**:
```
Entropy(S) = -Σ p_i log₂(p_i)
```

Convention: 0 log₂(0) = 0

**Information Gain**:
```
IG(S, A) = Entropy(S) - Σ (|S_v| / |S|) × Entropy(S_v)
```

where:
- `S` = dataset
- `A` = attribute to split on
- `S_v` = subset where attribute A = v
- `|S|` = number of samples

**Derivation from Information Theory**:
- Shannon entropy measures average surprise/uncertainty
- log₂(1/p_i) = bits needed to encode event with probability p_i
- Expected value: Σ p_i × log₂(1/p_i) = -Σ p_i log₂(p_i)

**Properties**:
- Range: [0, log₂(K)] where K = number of classes
- Entropy = 0: Pure node
- Entropy = log₂(K): Maximum uncertainty (uniform distribution)
- For binary: max entropy = 1 bit

**Example**:
```
Node with 7 Yes, 3 No:
Entropy = -0.7 log₂(0.7) - 0.3 log₂(0.3)
        = -0.7(-0.515) - 0.3(-1.737)
        = 0.360 + 0.521 = 0.881 bits

Pure node:
Entropy = -1.0 log₂(1.0) = 0 bits
```

#### Gini vs Entropy: Which to Use?

**Computational Cost**:
- Gini: Only squares and sums → faster
- Entropy: Requires logarithms → ~2x slower

**Behavior**:
- Both are concave functions that peak at 0.5 (binary case)
- Gini tends to isolate the most frequent class in its own branch
- Entropy tends to produce more balanced trees
- In practice: differences are minor (<2% accuracy difference)

**Recommendation**: Use Gini (sklearn default) unless you have a specific reason for entropy.

**Numerical Comparison** (binary classification):

| p(Yes) | Gini     | Entropy |
|--------|----------|---------|
| 0.0    | 0.000    | 0.000   |
| 0.1    | 0.180    | 0.469   |
| 0.3    | 0.420    | 0.881   |
| 0.5    | 0.500    | 1.000   |
| 0.7    | 0.420    | 0.881   |
| 0.9    | 0.180    | 0.469   |
| 1.0    | 0.000    | 0.000   |

### 2. Regression: Variance Reduction

#### Mean Squared Error (MSE)

**Definition**: Average squared deviation from the mean.

**Formula**:
```
MSE(S) = (1/n) Σ(y_i - ȳ)²
```

where:
- `y_i` = target value of sample i
- `ȳ` = mean of target values in set S
- `n` = number of samples

**Equivalent form** (variance):
```
Var(S) = MSE(S) = E[Y²] - E[Y]²
```

**For splitting**:
```
Weighted MSE = Σ (|S_v| / |S|) × MSE(S_v)
```

We choose the split that minimizes this weighted MSE.

**Why MSE?**:
- Differentiable (nice for optimization)
- Penalizes large errors more than small errors (quadratic)
- Related to L2 loss in linear regression
- Mathematically tractable

**Example**:
```
Node with values: [300k, 320k, 280k, 310k]
Mean = 302.5k

MSE = (1/4)[(300-302.5)² + (320-302.5)² + (280-302.5)² + (310-302.5)²]
    = (1/4)[6.25 + 306.25 + 506.25 + 56.25]
    = (1/4)[875]
    = 218.75

After split into [300k, 280k] and [320k, 310k]:
Left MSE = (1/2)[(300-290)² + (280-290)²] = 100
Right MSE = (1/2)[(320-315)² + (310-315)²] = 25
Weighted MSE = (2/4)×100 + (2/4)×25 = 62.5
Reduction = 218.75 - 62.5 = 156.25 ✓
```

#### Mean Absolute Error (MAE)

**Alternative to MSE**:
```
MAE(S) = (1/n) Σ|y_i - median(S)|
```

**Properties**:
- More robust to outliers than MSE
- Less commonly used (harder to optimize)
- sklearn: `criterion='absolute_error'` for regressors

**When to use MAE**:
- Dataset has extreme outliers
- Want predictions to be less influenced by rare extreme values

### 3. Splitting Criterion Comparison

| Criterion | Type          | Formula                  | Advantages                | Disadvantages              |
|-----------|---------------|--------------------------|---------------------------|----------------------------|
| Gini      | Classification| 1 - Σ(p_i)²             | Fast, isolates frequent   | May create unbalanced trees|
| Entropy   | Classification| -Σ p_i log₂(p_i)        | Balanced trees            | Slower (log computation)   |
| MSE       | Regression    | (1/n)Σ(y_i - ȳ)²        | Smooth, differentiable    | Sensitive to outliers      |
| MAE       | Regression    | (1/n)Σ\|y_i - median\|  | Robust to outliers        | Harder to optimize         |

---

## Decision Tree Algorithm Details

### CART (Classification and Regression Trees) Algorithm

**Pseudocode**:

```python
def build_tree(data, depth=0, max_depth=None, min_samples_split=2):
    """
    Recursively build a binary decision tree
    """
    n_samples = len(data)
    
    # Stopping criteria
    if (depth == max_depth or 
        n_samples < min_samples_split or 
        is_pure(data)):
        return create_leaf(data)
    
    # Find best split
    best_feature, best_threshold, best_gain = None, None, -inf
    
    for feature in features:
        for threshold in get_candidate_thresholds(data[feature]):
            gain = calculate_gain(data, feature, threshold)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    # No improvement possible
    if best_gain <= 0:
        return create_leaf(data)
    
    # Split and recurse
    left_data = data[data[best_feature] <= best_threshold]
    right_data = data[data[best_feature] > best_threshold]
    
    left_subtree = build_tree(left_data, depth+1, max_depth, min_samples_split)
    right_subtree = build_tree(right_data, depth+1, max_depth, min_samples_split)
    
    return Node(best_feature, best_threshold, left_subtree, right_subtree)
```

### Handling Categorical Features

**Two approaches**:

1. **Binary encoding** (One-Hot Encoding):
   - Convert each category into a binary feature
   - Example: Color ∈ {Red, Blue, Green} → is_Red, is_Blue, is_Green
   - Works with standard CART algorithm

2. **Multi-way splits**:
   - For feature with K categories, try all 2^(K-1) - 1 partitions
   - Computationally expensive for high-cardinality features
   - sklearn doesn't support this natively (use category_encoders library)

**Handling missing values**:
- Option 1: Drop rows with missing values
- Option 2: Impute with mean/median/mode before tree building
- Option 3: Surrogate splits (advanced, not in sklearn)
- Option 4: Treat missing as its own category

### Computational Complexity

**Training time**:
- Finding best split for one feature: O(n log n) for sorting
- Trying all features: O(d × n log n) where d = number of features
- Building entire tree: O(d × n log n × depth)
- Worst case depth: O(n) (degenerate tree)
- Balanced tree depth: O(log n)

**Overall**: O(d × n log n × log n) = O(d × n × log²n) for balanced trees

**Prediction time**:
- Traverse from root to leaf: O(depth)
- Balanced tree: O(log n)
- Worst case: O(n)

**Space complexity**:
- Storing tree: O(number of nodes)
- Balanced tree: O(2^depth) nodes
- Can be reduced by pruning

---

## Pruning Strategies

Overfitting is the main problem with decision trees. Pruning helps control model complexity.

### 1. Pre-Pruning (Early Stopping)

**Stop splitting when**:

- `max_depth` reached:
  ```python
  DecisionTreeClassifier(max_depth=5)
  ```
  - Typical values: 3-10 for interpretable models, 10-20 for ensembles
  - Too low: underfits, too high: overfits

- `min_samples_split` threshold:
  ```python
  DecisionTreeClassifier(min_samples_split=20)
  ```
  - Minimum samples required to split a node
  - Typical: 2-50

- `min_samples_leaf` threshold:
  ```python
  DecisionTreeClassifier(min_samples_leaf=10)
  ```
  - Minimum samples required in a leaf
  - Prevents tiny leaves that overfit

- `max_leaf_nodes` limit:
  ```python
  DecisionTreeClassifier(max_leaf_nodes=50)
  ```
  - Caps total number of leaves
  - Uses best-first search instead of depth-first

- `min_impurity_decrease`:
  ```python
  DecisionTreeClassifier(min_impurity_decrease=0.01)
  ```
  - Split must reduce impurity by at least this amount
  - Filters out weak splits

**Pros**: Fast, simple, happens during training
**Cons**: May stop too early (greedy decision)

### 2. Post-Pruning (Cost-Complexity Pruning)

**Build full tree, then prune back**:

**Cost-Complexity criterion**:
```
R_α(T) = R(T) + α|T|
```

where:
- `R(T)` = error rate of tree T on validation set
- `|T|` = number of leaves
- `α` = complexity parameter (higher α → more pruning)

**Algorithm** (sklearn's `ccp_alpha`):

1. Build full tree
2. For each α value, find the smallest subtree that minimizes R_α(T)
3. Evaluate on validation set
4. Choose α that gives best validation performance

```python
from sklearn.tree import DecisionTreeClassifier

# Get pruning path
clf = DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Try different alphas
best_alpha = None
best_score = 0

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    
    if score > best_score:
        best_score = score
        best_alpha = alpha

# Train final model
final_clf = DecisionTreeClassifier(ccp_alpha=best_alpha)
final_clf.fit(X_train, y_train)
```

**Pros**: More principled, can achieve better bias-variance tradeoff
**Cons**: Slower (requires validation set and multiple training runs)

### Reduced Error Pruning (REP)

**Simple alternative**:

1. Build full tree on training set
2. For each internal node, from bottom up:
   - Try replacing the subtree with a leaf
   - If validation accuracy improves (or stays same), prune
3. Continue until no more improvements

**Implementation sketch**:
```python
def reduced_error_pruning(tree, X_val, y_val):
    """
    Prune tree based on validation performance
    """
    improved = True
    
    while improved:
        improved = False
        baseline_score = tree.score(X_val, y_val)
        
        for node in tree.get_internal_nodes():
            # Try pruning this subtree
            original_subtree = node.copy()
            node.convert_to_leaf()
            
            new_score = tree.score(X_val, y_val)
            
            if new_score >= baseline_score:
                # Keep it pruned
                improved = True
                baseline_score = new_score
            else:
                # Restore original subtree
                node.restore(original_subtree)
    
    return tree
```

---

## Random Forest Deep Dive

### 1. Bootstrap Aggregating (Bagging)

**Core idea**: Train multiple models on different random subsets and average their predictions.

**Bootstrap sampling**:
```python
def bootstrap_sample(data, n_samples=None):
    """
    Sample with replacement from data
    """
    if n_samples is None:
        n_samples = len(data)
    
    indices = np.random.randint(0, len(data), n_samples)
    return data[indices]
```

**Key insight**: Each bootstrap sample contains ~63.2% unique samples.

**Derivation**:
- Probability of NOT being selected in one draw: (n-1)/n
- Probability of NOT being selected in n draws: ((n-1)/n)^n
- As n → ∞: ((n-1)/n)^n → 1/e ≈ 0.368
- Therefore, probability of being selected: 1 - 1/e ≈ 0.632

### 2. Random Feature Selection

**At each split**:
- Don't consider all d features
- Randomly select m features to consider
- Choose best split among those m features

**Typical values**:
- Classification: m = √d (sklearn default: `max_features='sqrt'`)
- Regression: m = d/3 (sklearn default: `max_features=1.0/3.0`)

**Why does this help?**:
- Decorrelates trees
- If one feature is very strong, bagging alone would create similar trees
- Random features forces diversity

**Example**:
```
Dataset with 100 features:
- Classification RF: each split considers √100 = 10 random features
- Regression RF: each split considers 100/3 ≈ 33 random features
```

### 3. Out-of-Bag (OOB) Error

**Free validation estimate**:

Since each tree is trained on ~63% of data, the remaining ~37% can be used for validation.

**OOB Score calculation**:
```python
def calculate_oob_score(forest, X, y):
    """
    Calculate out-of-bag score for random forest
    """
    n_samples = len(X)
    oob_predictions = np.zeros(n_samples)
    oob_counts = np.zeros(n_samples)
    
    for tree, bootstrap_indices in zip(forest.trees, forest.bootstrap_indices):
        # OOB samples for this tree
        oob_mask = ~np.isin(np.arange(n_samples), bootstrap_indices)
        oob_indices = np.where(oob_mask)[0]
        
        # Predict on OOB samples
        predictions = tree.predict(X[oob_indices])
        
        # Accumulate
        oob_predictions[oob_indices] += predictions
        oob_counts[oob_indices] += 1
    
    # Average predictions
    oob_predictions = oob_predictions / oob_counts
    
    # Calculate accuracy
    return accuracy_score(y, oob_predictions)
```

**Usage in sklearn**:
```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_}")
```

**Advantages**:
- No need for separate validation set
- More data available for training
- Good estimate of generalization error

### 4. Feature Importance

**Two methods in sklearn**:

**Method 1: Impurity-based (default)**:
```python
importance_i = Σ (weighted impurity decrease at splits using feature i) / Σ (all impurity decreases)
```

- Fast to compute
- Available immediately after training
- Biased toward high-cardinality features

**Method 2: Permutation-based**:
```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_test, y_test, n_repeats=10)
importances = result.importances_mean
```

- Shuffle feature i, measure drop in accuracy
- More reliable but slower
- Works with any model

**Usage**:
```python
import matplotlib.pyplot as plt

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
```

### 5. When Random Forest Fails

**Not suitable for**:
- High-dimensional sparse data (text with TF-IDF)
- Small datasets (< 1000 samples)
- Linear relationships (use linear model instead)
- Extrapolation (RFs can't predict beyond training range)

**Example of extrapolation failure**:
```python
# Train on data where target is in range [0, 100]
X_train = np.linspace(0, 10, 100).reshape(-1, 1)
y_train = 10 * X_train.flatten()

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Test on data where target should be 150
X_test = np.array([[15]])
print(rf.predict(X_test))  # Will predict ~100, not 150!
```

Trees can only output values they've seen in training leaves.

---

## Gradient Boosting Internals

### 1. Boosting Framework

**Key idea**: Build trees sequentially, each correcting errors of the previous ensemble.

**General framework**:
```
F₀(x) = initial guess (e.g., mean of y)

For m = 1 to M:
    1. Calculate residuals: r_i = y_i - F_{m-1}(x_i)
    2. Fit tree h_m to residuals
    3. Update: F_m(x) = F_{m-1}(x) + ν × h_m(x)
    
Final prediction: F_M(x)
```

where ν is the learning rate.

### 2. Gradient Boosting for Regression

**Mathematical formulation**:

Minimize loss function:
```
L(y, F(x)) = Σ ℓ(y_i, F(x_i))
```

For MSE: ℓ(y, F(x)) = (y - F(x))²

**Gradient descent in function space**:

The negative gradient of MSE:
```
-∂L/∂F(x_i) = 2(y_i - F(x_i)) = 2 × residual
```

So fitting to residuals = gradient descent!

**Algorithm**:
```python
def gradient_boosting_regressor(X, y, n_trees, learning_rate, max_depth):
    """
    Gradient boosting for regression
    """
    # Initialize with mean
    F = np.full(len(y), np.mean(y))
    trees = []
    
    for m in range(n_trees):
        # Calculate negative gradient (residuals)
        residuals = y - F
        
        # Fit tree to residuals
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X, residuals)
        
        # Update predictions
        F += learning_rate * tree.predict(X)
        
        trees.append(tree)
    
    return trees

def predict(X, trees, learning_rate, initial_value):
    """
    Make predictions with gradient boosting model
    """
    F = np.full(len(X), initial_value)
    
    for tree in trees:
        F += learning_rate * tree.predict(X)
    
    return F
```

### 3. Gradient Boosting for Classification

**Binary classification with log loss**:

Loss function:
```
ℓ(y, F(x)) = log(1 + exp(-y × F(x)))
```

where y ∈ {-1, +1} and F(x) is the raw score (before sigmoid).

**Gradient**:
```
-∂ℓ/∂F(x) = y / (1 + exp(y × F(x)))
```

**Pseudo-residuals**:
```
r_i = y_i / (1 + exp(y_i × F(x_i)))
```

**Convert to probability**:
```
P(y=1|x) = 1 / (1 + exp(-F(x)))
```

### 4. XGBoost Improvements

**Key innovations**:

1. **Second-order approximation**:
   - Uses both gradient and Hessian (second derivative)
   - More accurate updates

2. **Regularization**:
   ```
   Obj = Σ ℓ(y_i, F(x_i)) + Σ Ω(f_m)
   
   where Ω(f) = γT + (λ/2)||w||²
   ```
   - γ: penalty for number of leaves T
   - λ: L2 penalty on leaf weights

3. **Column (feature) sampling**:
   - Like RF, subsample features per tree
   - Reduces overfitting and speeds up training

4. **Sparsity-aware split finding**:
   - Handles missing values efficiently
   - Learns best direction for missing values

5. **Approximate split finding**:
   - Quantile sketch algorithm for large datasets
   - Bins continuous features

**Hyperparameters**:
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,        # Number of trees
    learning_rate=0.1,       # η (eta): shrinkage
    max_depth=6,             # Maximum tree depth
    min_child_weight=1,      # Minimum sum of instance weights in leaf
    gamma=0,                 # Minimum loss reduction for split
    subsample=0.8,           # Row sampling per tree
    colsample_bytree=0.8,    # Feature sampling per tree
    reg_alpha=0,             # L1 regularization
    reg_lambda=1,            # L2 regularization
)
```

### 5. LightGBM Improvements

**Key differences from XGBoost**:

1. **Leaf-wise growth** (instead of level-wise):
   - Splits leaf with max delta loss
   - Faster convergence
   - More prone to overfitting (use max_depth)

2. **Gradient-based One-Side Sampling (GOSS)**:
   - Keep all samples with large gradients
   - Random sample from small gradients
   - Speeds up training on large datasets

3. **Exclusive Feature Bundling (EFB)**:
   - Bundle mutually exclusive features
   - Reduces number of features without losing information

**Usage**:
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,            # No limit (uses num_leaves instead)
    num_leaves=31,           # Max leaves per tree
    min_data_in_leaf=20,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

---

## Hyperparameter Tuning Guide

### Decision Trees

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|--------|
| max_depth | Maximum tree depth | 3-20 | ↑ depth → ↑ overfit |
| min_samples_split | Min samples to split | 2-50 | ↑ → ↓ overfit |
| min_samples_leaf | Min samples in leaf | 1-50 | ↑ → ↓ overfit |
| max_features | Features to consider | √d or log₂(d) | ↓ → ↓ overfit |
| max_leaf_nodes | Max leaves | 10-100 | ↓ → ↓ overfit |
| ccp_alpha | Pruning parameter | 0.0-0.1 | ↑ → ↓ overfit |

**Tuning strategy**:
1. Start with default (no limits)
2. If overfitting: add max_depth=5, min_samples_split=20
3. Fine-tune with grid search

### Random Forest

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|--------|
| n_estimators | Number of trees | 100-1000 | ↑ better (but slower) |
| max_depth | Tree depth | 10-50 or None | ↑ → ↑ overfit per tree |
| min_samples_split | Min samples to split | 2-20 | ↑ → ↓ overfit |
| max_features | Features per split | √d, log₂(d), 0.3 | Diversity vs accuracy |
| bootstrap | Use bootstrap | True | False = pasting |
| oob_score | Calculate OOB | True | Free validation |

**Tuning strategy**:
1. Start with n_estimators=100, defaults
2. Increase n_estimators until OOB score plateaus
3. Tune max_depth and min_samples_split if needed
4. Try different max_features values

**Example**:
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'max_features': ['sqrt', 'log2', 0.3]
}

rf = RandomForestClassifier(oob_score=True)
search = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, n_jobs=-1)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best CV score: {search.best_score_}")
```

### XGBoost

| Parameter | Description | Typical Range | Effect |
|-----------|-------------|---------------|--------|
| n_estimators | Number of trees | 100-1000 | ↑ better (watch for overfit) |
| learning_rate | Shrinkage | 0.01-0.3 | ↓ → need more trees |
| max_depth | Tree depth | 3-10 | ↑ → ↑ overfit |
| min_child_weight | Min weight in leaf | 1-10 | ↑ → ↓ overfit |
| gamma | Min loss reduction | 0-5 | ↑ → ↓ overfit |
| subsample | Row sampling | 0.5-1.0 | ↓ → ↓ overfit |
| colsample_bytree | Feature sampling | 0.5-1.0 | ↓ → ↓ overfit |
| reg_alpha | L1 regularization | 0-1 | ↑ → ↓ overfit |
| reg_lambda | L2 regularization | 0-1 | ↑ → ↓ overfit |

**Tuning strategy**:
1. Fix learning_rate=0.1, tune n_estimators with early stopping
2. Tune max_depth and min_child_weight
3. Tune gamma
4. Tune subsample and colsample_bytree
5. Tune regularization (reg_alpha, reg_lambda)
6. Lower learning_rate and increase n_estimators

**Example with early stopping**:
```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 6,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=False
)

print(f"Best iteration: {model.best_iteration}")
```

---

## Advanced Topics

### 1. Imbalanced Classification

**Problem**: When one class is much more frequent than others (e.g., 95% negative, 5% positive).

**Solutions**:

**Option 1: Class weights**:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weight_dict = dict(zip(np.unique(y), class_weights))

rf = RandomForestClassifier(class_weight=weight_dict)
rf.fit(X_train, y_train)
```

**Option 2: Balanced Random Forest**:
```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(n_estimators=100)
brf.fit(X_train, y_train)
```

**Option 3: Threshold tuning**:
```python
# Get probability predictions
y_proba = rf.predict_proba(X_test)[:, 1]

# Try different thresholds
from sklearn.metrics import f1_score

best_threshold = 0.5
best_f1 = 0

for threshold in np.linspace(0.1, 0.9, 17):
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}")
```

### 2. Multi-output Regression/Classification

**Problem**: Predict multiple targets simultaneously.

**Example**:
```python
# Predict [price, sqft, bedrooms] from location data
X = location_features
y = np.column_stack([prices, sqfts, bedrooms])

rf = RandomForestRegressor()
rf.fit(X, y)

predictions = rf.predict(X_test)  # Shape: (n_samples, 3)
```

**Note**: Each target gets its own set of trees, but they share the same structure.

### 3. Extremely Randomized Trees (Extra Trees)

**Difference from Random Forest**:
- RF: Bootstrap sample + find best split among random features
- Extra Trees: Whole dataset + random split thresholds

**Advantage**: Faster training (no sorting needed)
**Disadvantage**: Slightly worse performance

```python
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_estimators=100)
et.fit(X_train, y_train)
```

### 4. Calibration

**Problem**: Tree-based models' probability estimates can be poorly calibrated.

**Solution**: Platt scaling or isotonic regression

```python
from sklearn.calibration import CalibratedClassifierCV

rf = RandomForestClassifier(n_estimators=100)
calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=5)
calibrated_rf.fit(X_train, y_train)

# Now probabilities are better calibrated
probs = calibrated_rf.predict_proba(X_test)
```

### 5. Partial Dependence Plots

**Visualize feature effect on predictions**:

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(
    rf, X, features=[0, 1, (0, 1)], 
    feature_names=feature_names,
    ax=ax
)
plt.tight_layout()
plt.show()
```

Shows how predictions change as feature values change, averaging over all other features.

### 6. SHAP Values

**More sophisticated feature importance**:

```python
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)

# Force plot for single prediction
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test[0])
```

SHAP (SHapley Additive exPlanations) provides game-theoretic feature attributions.

---

## Coding Assignment

### Part 1: Implement Decision Tree from Scratch

**Objective**: Build a binary classification decision tree with Gini impurity.

**Requirements**:
1. Implement `calculate_gini(y)` function
2. Implement `find_best_split(X, y)` function
3. Implement `Node` and `Leaf` classes
4. Implement `build_tree(X, y, depth, max_depth)` recursive function
5. Implement `predict(tree, x)` function
6. Test on Iris dataset

**Starter code**:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, feature_idx, threshold, left, right):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

class Leaf:
    def __init__(self, value):
        self.value = value  # Class label (most common in node)

def calculate_gini(y):
    """
    Calculate Gini impurity of a dataset
    
    Args:
        y: array of labels
    
    Returns:
        float: Gini impurity
    """
    # TODO: Implement this
    pass

def find_best_split(X, y):
    """
    Find the best feature and threshold to split on
    
    Args:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
    
    Returns:
        best_feature_idx: index of best feature to split on
        best_threshold: best threshold value
        best_gini: Gini impurity of the split
    """
    # TODO: Implement this
    # Hint: Try all features and all unique values as thresholds
    pass

def build_tree(X, y, depth=0, max_depth=5):
    """
    Recursively build decision tree
    
    Args:
        X: feature matrix
        y: labels
        depth: current depth
        max_depth: maximum depth allowed
    
    Returns:
        Node or Leaf
    """
    # TODO: Implement this
    # Base cases:
    #   1. depth >= max_depth
    #   2. All samples have same label (pure node)
    #   3. Cannot find a good split
    # Recursive case:
    #   1. Find best split
    #   2. Split data
    #   3. Recursively build left and right subtrees
    pass

def predict_one(tree, x):
    """
    Predict single sample
    
    Args:
        tree: root node of tree
        x: single sample
    
    Returns:
        prediction (class label)
    """
    # TODO: Implement this
    # Hint: Traverse tree until reaching a leaf
    pass

def predict(tree, X):
    """
    Predict multiple samples
    """
    return np.array([predict_one(tree, x) for x in X])

# Test your implementation
iris = load_iris()
X = iris.data
y = iris.target

# Use only 2 classes for binary classification
mask = y < 2
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build your tree
my_tree = build_tree(X_train, y_train, max_depth=5)

# Make predictions
y_pred = predict(my_tree, X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Your tree accuracy: {accuracy:.4f}")
```

**Expected output**: Your tree should achieve >90% accuracy on Iris binary classification.

### Part 2: Compare with sklearn

**Task**: Train sklearn's DecisionTreeClassifier on the same data and compare.

```python
from sklearn.tree import DecisionTreeClassifier

# Train sklearn tree
sklearn_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
sklearn_tree.fit(X_train, y_train)

# Predict
sklearn_pred = sklearn_tree.predict(X_test)

# Compare
sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
print(f"sklearn tree accuracy: {sklearn_accuracy:.4f}")
print(f"Difference: {abs(accuracy - sklearn_accuracy):.4f}")
```

**Questions to answer**:
1. Why might your implementation differ from sklearn's?
2. What features does sklearn have that yours doesn't?

### Part 3: Build Random Forest

**Task**: Implement a simple Random Forest by combining multiple trees.

```python
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
    
    def fit(self, X, y):
        """
        Train random forest
        
        Requirements:
        1. For each tree:
           - Create bootstrap sample
           - Randomly select features (use self.max_features)
           - Build tree on bootstrapped data
        2. Store all trees
        """
        # TODO: Implement this
        pass
    
    def predict(self, X):
        """
        Predict using majority vote
        
        Requirements:
        1. Get predictions from all trees
        2. Return majority vote for each sample
        """
        # TODO: Implement this
        pass

# Test your Random Forest
my_rf = RandomForest(n_trees=50, max_depth=5, max_features=int(np.sqrt(X.shape[1])))
my_rf.fit(X_train, y_train)

rf_pred = my_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Your Random Forest accuracy: {rf_accuracy:.4f}")

# Compare with sklearn
from sklearn.ensemble import RandomForestClassifier

sklearn_rf = RandomForestClassifier(n_estimators=50, max_depth=5, max_features='sqrt', random_state=42)
sklearn_rf.fit(X_train, y_train)

sklearn_rf_pred = sklearn_rf.predict(X_test)
sklearn_rf_accuracy = accuracy_score(y_test, sklearn_rf_pred)

print(f"sklearn Random Forest accuracy: {sklearn_rf_accuracy:.4f}")
```

**Expected result**: Your Random Forest should achieve >95% accuracy.

### Part 4: Visualize Decision Boundaries

**Task**: Visualize how trees partition the feature space.

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, title):
    """
    Plot decision boundary for 2D data
    """
    h = 0.02  # Step size
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

# Use only first 2 features for visualization
X_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

# Build trees on 2D data
single_tree = build_tree(X_2d, y_train, max_depth=3)
my_rf_2d = RandomForest(n_trees=50, max_depth=3)
my_rf_2d.fit(X_2d, y_train)

# Plot
plot_decision_boundary(lambda x: predict(single_tree, x), X_2d, y_train, 
                       'Single Decision Tree (depth=3)')
plot_decision_boundary(my_rf_2d, X_2d, y_train, 
                       'Random Forest (50 trees, depth=3)')
```

**Observations to make**:
1. How do single tree boundaries look? (Rectangular, axis-aligned)
2. How does RF smooth the boundaries?
3. Where does each model overfit or underfit?

### Bonus Challenge 1: Feature Importance

**Task**: Implement feature importance calculation.

```python
def calculate_feature_importance(tree, n_features):
    """
    Calculate feature importance based on total impurity decrease
    
    Returns:
        importance: array of shape (n_features,)
    """
    # TODO: Implement this
    # Hint: Traverse tree, sum up Gini decrease for each feature
    pass
```

### Bonus Challenge 2: XGBoost from Scratch

**Task**: Implement a simplified gradient boosting for regression.

```python
class GradientBoostingRegressor:
    def __init__(self, n_trees=100, learning_rate=0.1, max_depth=3):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X, y):
        """
        Implement gradient boosting
        
        Algorithm:
        1. Initialize F with mean of y
        2. For each tree:
           - Calculate residuals: y - F
           - Fit tree to residuals
           - Update F: F += learning_rate * tree.predict(X)
        """
        # TODO: Implement this
        pass
    
    def predict(self, X):
        """
        Sum predictions from all trees
        """
        # TODO: Implement this
        pass

# Test on regression problem
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2)

my_gb = GradientBoostingRegressor(n_trees=100, learning_rate=0.1, max_depth=3)
my_gb.fit(X_train_reg, y_train_reg)

from sklearn.metrics import mean_squared_error
y_pred_gb = my_gb.predict(X_test_reg)
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_gb):.2f}")
```

### Submission Guidelines

1. **Code**: Submit a Jupyter notebook or Python script with all implementations
2. **Report**: Include a PDF with:
   - Accuracy comparisons
   - Decision boundary plots
   - Discussion of differences between your implementation and sklearn
   - Feature importance analysis
3. **Bonus**: If you complete bonus challenges, include them in the notebook
(All of these are optional but encouraged provided you have time for them)
---

## Additional Resources

### Books
1. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
   - Chapter 9: Additive Models, Trees, and Related Methods
   - Chapter 10: Boosting and Additive Trees
   - Free PDF: https://web.stanford.edu/~hastie/ElemStatLearn/

2. **"Pattern Recognition and Machine Learning"** - Bishop
   - Chapter 14.4: Classification and Regression Trees

3. **"Hands-On Machine Learning"** - Géron
   - Chapter 6: Decision Trees
   - Chapter 7: Ensemble Learning and Random Forests

### Papers

1. **CART Original Paper**:
   - Breiman et al. (1984) "Classification and Regression Trees"

2. **Random Forests**:
   - Breiman (2001) "Random Forests" - Machine Learning 45(1):5-32

3. **XGBoost**:
   - Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System" - KDD

4. **LightGBM**:
   - Ke et al. (2017) "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" - NIPS

### Online Resources

1. **sklearn Documentation**:
   - Decision Trees: https://scikit-learn.org/stable/modules/tree.html
   - Ensembles: https://scikit-learn.org/stable/modules/ensemble.html

2. **XGBoost Documentation**:
   - https://xgboost.readthedocs.io/

3. **StatQuest Videos** (Josh Starmer):
   - Decision Trees: https://www.youtube.com/watch?v=7VeUPuFGJHk
   - Random Forests: https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
   - Gradient Boost: https://www.youtube.com/watch?v=3CC4N4z3GJc

4. **Kaggle Tutorials**:
   - Feature Engineering for Trees
   - Hyperparameter Tuning for XGBoost

### Datasets for Practice

1. **UCI Machine Learning Repository**:
   - https://archive.ics.uci.edu/ml/index.php
   - Try: Adult Income, Wine Quality, Bank Marketing

2. **Kaggle Datasets**:
   - Titanic (classification)
   - House Prices (regression)
   - Credit Card Fraud (imbalanced classification)

3. **sklearn Built-in**:
   - Iris, Wine, Breast Cancer (classification)
   - Boston Housing, Diabetes (regression)

---

## Frequently Asked Questions

**Q: When should I use Random Forest vs XGBoost?**
A: Quick guideline:
- Start with Random Forest: Easier to tune, robust, good baseline
- Use XGBoost when: You need that extra 2-5% accuracy, have time to tune, competing in Kaggle

**Q: How many trees should I use?**
A: 
- Random Forest: 100-500 usually sufficient. More doesn't hurt (just slower).
- XGBoost: Use early stopping. Usually 100-1000 with learning_rate=0.1.

**Q: My tree overfits even with max_depth=3. What should I do?**
A:
1. Check for data leakage
2. Increase min_samples_leaf
3. Try feature selection
4. Use ensemble (RF/XGBoost) instead of single tree

**Q: Can trees handle missing values?**
A:
- sklearn: No, must impute first
- XGBoost/LightGBM: Yes, they learn best direction for missing values

**Q: Why is my Random Forest worse than a single tree?**
A: Usually means:
- Too few trees (increase n_estimators)
- max_features too small (trees are too similar)
- Data is very small (<100 samples)

**Q: When should I NOT use trees?**
A:
- High-dimensional sparse data (text, genomics) → Linear models, deep learning
- Need extrapolation → Linear regression
- Need well-calibrated probabilities → Logistic regression, calibrated models
- Very smooth relationships → Neural networks, polynomial regression

**Q: How do I handle categorical features?**
A:
- sklearn: One-hot encode before fitting
- XGBoost: Can handle `enable_categorical=True` (experimental)
- LightGBM: Native categorical support with `categorical_feature` parameter

---

## Summary Cheat Sheet

### Quick Decision Guide

```
Classification task?
├─ Yes
│  ├─ Need interpretability?
│  │  └─ Yes → Single Decision Tree (max_depth=5-10)
│  └─ No
│     ├─ Quick baseline needed?
│     │  └─ Yes → Random Forest (n_estimators=100)
│     └─ No, need best accuracy
│        └─ XGBoost with hyperparameter tuning
│
└─ No (Regression)
   ├─ Need interpretability?
   │  └─ Yes → Single Decision Tree
   └─ No
      ├─ Linear relationship?
      │  └─ Yes → Linear Regression
      └─ No
         ├─ Quick baseline?
         │  └─ Yes → Random Forest
         └─ Need extrapolation?
            ├─ Yes → NOT trees (linear model or neural net)
            └─ No → XGBoost
```

### Default Hyperparameters to Start With

```python
# Single Tree (for interpretation)
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10
)

# Random Forest (quick baseline)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    max_features='sqrt',
    oob_score=True
)

# XGBoost (for competitions)
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8
)
```

### Common Pitfalls to Avoid

1. Forgetting to handle missing values for sklearn
2. Not checking for data leakage
3. Using trees for high-dimensional sparse data
4. Expecting trees to extrapolate beyond training range
5. Not using early stopping with gradient boosting
6. Tuning too many hyperparameters at once
7. Ignoring class imbalance
8. Using default parameters for XGBoost in production

---

**End of Reading Material**

For questions or clarifications, reach out on the me on WhatsApp(8826350047).

Good luck with the assignment! 🌲🚀
