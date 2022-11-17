### Finetuning Hyperparameters of Logistic regression ML Algorithm
The basic Logisitic Regression model is a supervised classification ML algorithm, that ideally works on binary classification problems.
There are various hyperparameters that can be modified in order to fine tune the model performance and obtain the best possible results.
List of hyperparameters used in the below script are:
- Penalty: In order to optmize the performance of the model and trace the important features different penalties can be employed. Lasso (L1) , Ridge (L2) and ElasticNet are the three types of penalties that can be used.
- Solver: Since Logisitic Regression algorithm works on optimization technique, various optmization methods are available to be used in the model. The selection of right optimizer solver depends on the penalty. Different optmizer solvers are compatible with different penalty types and hence proper selection of the optimizer solver is important.
- C: C is called the regularization parameter. Technically C is inverse of the penalty term. But an easier way to understand C is to relate it to the model complexity. C denotes the model complexity. Smaller values of C indicate a simple model and larger values of C indicate complex models. Selecting the optimum value of C to create a balanced model is must.
- max_iter: This parameter specifies the maximum number of iterations required to reach the minima. If the number of iterations needed to converge is higher than the max_iter provided the model would fail to converge and hence the true minima of the loss value is not achieved. This hinders the model in achieving the best possible performance. Hence providing a wide range of max_iter values helps the model to achieve better performance.

#### Installing Dependencies


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')
```




    sklearn.exceptions.ConvergenceWarning('ignore')



#### Data Ingestion


```python
bio = pd.read_csv('healthcare_data.csv')
bio.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>...</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
      <th>Unnamed: 32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302</td>
      <td>M</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517</td>
      <td>M</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300903</td>
      <td>M</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348301</td>
      <td>M</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358402</td>
      <td>M</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
bio.shape
```




    (569, 33)




```python
bio.columns
```




    Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
           'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
           'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
           'fractal_dimension_se', 'radius_worst', 'texture_worst',
           'perimeter_worst', 'area_worst', 'smoothness_worst',
           'compactness_worst', 'concavity_worst', 'concave points_worst',
           'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'],
          dtype='object')




```python
bio.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 33 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   id                       569 non-null    int64  
     1   diagnosis                569 non-null    object 
     2   radius_mean              569 non-null    float64
     3   texture_mean             569 non-null    float64
     4   perimeter_mean           569 non-null    float64
     5   area_mean                569 non-null    float64
     6   smoothness_mean          569 non-null    float64
     7   compactness_mean         569 non-null    float64
     8   concavity_mean           569 non-null    float64
     9   concave points_mean      569 non-null    float64
     10  symmetry_mean            569 non-null    float64
     11  fractal_dimension_mean   569 non-null    float64
     12  radius_se                569 non-null    float64
     13  texture_se               569 non-null    float64
     14  perimeter_se             569 non-null    float64
     15  area_se                  569 non-null    float64
     16  smoothness_se            569 non-null    float64
     17  compactness_se           569 non-null    float64
     18  concavity_se             569 non-null    float64
     19  concave points_se        569 non-null    float64
     20  symmetry_se              569 non-null    float64
     21  fractal_dimension_se     569 non-null    float64
     22  radius_worst             569 non-null    float64
     23  texture_worst            569 non-null    float64
     24  perimeter_worst          569 non-null    float64
     25  area_worst               569 non-null    float64
     26  smoothness_worst         569 non-null    float64
     27  compactness_worst        569 non-null    float64
     28  concavity_worst          569 non-null    float64
     29  concave points_worst     569 non-null    float64
     30  symmetry_worst           569 non-null    float64
     31  fractal_dimension_worst  569 non-null    float64
     32  Unnamed: 32              0 non-null      float64
    dtypes: float64(31), int64(1), object(1)
    memory usage: 146.8+ KB



```python
bio.diagnosis.unique()
```




    array(['M', 'B'], dtype=object)



#### Brief description of the data
- Total 569 records of patients and 33 columns
- id is a unique column that does not contribue to prediction
- unnamed: 32 is a column with no value in it and hence cannot contribute
- diagnosis is an object type feature with only two values: M and B
- diagnosis is the target variable with two categories: M and B
- Remaining all the columns are float type and act as predictor variables

#### Basic Preprocessing


```python
# Remove features 'id' & 'Unnamed: 32' since they do not help in predicting
bio = bio.drop(['id', 'Unnamed: 32'],axis=1)
```


```python
# Encoding target variable 'diagnosis'
bio['diagnosis'] = bio['diagnosis'].map({'M': 0,'B': 1})

#Encoding: a preprocessing technique used to convert categorical variables to number codes.
```


```python
bio.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diagnosis</th>
      <th>radius_mean</th>
      <th>texture_mean</th>
      <th>perimeter_mean</th>
      <th>area_mean</th>
      <th>smoothness_mean</th>
      <th>compactness_mean</th>
      <th>concavity_mean</th>
      <th>concave points_mean</th>
      <th>symmetry_mean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>perimeter_worst</th>
      <th>area_worst</th>
      <th>smoothness_worst</th>
      <th>compactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>symmetry_worst</th>
      <th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
bio.shape
```




    (569, 31)



#### Above dataset is finally transformed with all numerical values. 30 predictor variables and 1 target variable.

### Splitting the dataset into X and y


```python
X = bio.drop(['diagnosis'],axis = 1)
y = bio['diagnosis']
```


```python
X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=32)
```

#### Logistic regression is a distance based model, hence all numerical features must be scaled.


```python
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
x_test = ss.transform(x_test)
```

### Basic model training and inferencing

#### Instantiation and training


```python
lr = LogisticRegression(random_state=10)
model = lr.fit(X_train, y_train)
```

#### Inferencing


```python
y_pred = model.predict(x_test)
```

#### Evaluation


```python
print(accuracy_score(y_test,y_pred))
```

    0.9578947368421052



```python
print(model.get_params)
```

    <bound method BaseEstimator.get_params of LogisticRegression(random_state=10)>


### Hyperparameters Finetuning
#### Hyperparameters used:
- penalty: L1 , L2 and elasticnet. By default penalty is L2.
- C: regularization parameter. This depicts the complexity of model
- solver: Optimization problems can be solved by different solvers
- max_iters: in how many steps the global minima is reached


```python
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
```


```python
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear'],
    'max_iter' : [500, 1000, 1500]
    }
]

3*20*5*4
```




    1200




```python
np.logspace(-4,4,20)   #.0001 to 10000
```




    array([1.00000000e-04, 2.63665090e-04, 6.95192796e-04, 1.83298071e-03,
           4.83293024e-03, 1.27427499e-02, 3.35981829e-02, 8.85866790e-02,
           2.33572147e-01, 6.15848211e-01, 1.62377674e+00, 4.28133240e+00,
           1.12883789e+01, 2.97635144e+01, 7.84759970e+01, 2.06913808e+02,
           5.45559478e+02, 1.43844989e+03, 3.79269019e+03, 1.00000000e+04])



#### Comments:
- It is just way to give range of values that c can take.
- You may also use the simple approach as shown below:
- 'C' : [.0001, .001, .01, .1, 1, 10, 100, 1000, 10000]


```python
import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)
```


```python
gsc = GridSearchCV(lr, param_grid = param_grid, cv = 10,verbose=True, n_jobs=-1)
tuned_model = gsc.fit(X_train,y_train)
```

    Fitting 10 folds for each of 540 candidates, totalling 5400 fits



```python
tuned_model.best_estimator_
```




    LogisticRegression(C=0.615848211066026, max_iter=500, random_state=10)




```python
print (f'Accuracy - : {tuned_model.score(x_test,y_test):.3f}')
```

    Accuracy - : 0.963



```python
lr_finetuned = lr.set_params(C = 0.615848211066026, 
                             max_iter = 500,
                             random_state = 10)
```


```python
model_final = lr_finetuned.fit(X_train, y_train)
y_pred = model_final.predict(x_test)
print(accuracy_score(y_test,y_pred))
```

    0.9631578947368421



```python
model_final.get_params
```




    <bound method BaseEstimator.get_params of LogisticRegression(C=0.615848211066026, max_iter=500, random_state=10)>




```python
model_final.penalty
```




    'l2'




```python
model_final.coef_
```




    array([[-0.48507112, -0.39769552, -0.44891167, -0.5166906 , -0.25795931,
             0.22433419, -0.59320734, -0.65494822,  0.14756751,  0.20464987,
            -1.03507872, -0.18521633, -0.58477086, -0.78295621, -0.05402549,
             0.45576588, -0.01313436, -0.05961217,  0.47307332,  0.38539697,
            -0.87864902, -0.94643093, -0.75975075, -0.82213525, -0.58373362,
            -0.03479723, -0.57764982, -0.5404704 , -0.51782045, -0.29460082]])



#### Trying working with L1


```python
model_l1 = LogisticRegression(penalty='l1',C=1, max_iter =10, random_state=10, solver = 'liblinear' )
```


```python
model_l1_fit = model_l1.fit(X_train,y_train)
```


```python
y_pred_l1 = model_l1.predict(x_test)
```


```python
accuracy_score(y_test,y_pred_l1)
```




    0.9526315789473684




```python
model_l1_fit.coef_
```




    array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        , -0.25240274, -0.8403526 ,  0.        ,  0.        ,
            -2.41841316,  0.        ,  0.        , -0.02519683,  0.        ,
             0.27464826,  0.        ,  0.        ,  0.27932633,  0.21351214,
            -1.55360797, -1.56797121, -0.47396785, -3.07790957, -0.76127109,
             0.        , -0.59962018, -0.4315926 , -0.21474998,  0.        ]])



### Observations:
- The highest accuracy is obtained after finetuning the model.
- Base model accuracy: 95.79%; Fine-tuned model accuracy: 96.32%

- With L1 penalty specific experiment is done just to verify the results.
- The coefficients of L1 model show how many feature coefficients are equated to zero.
- So L1 does help in feature selection. All non-zero coefficient features are considered for final model.
