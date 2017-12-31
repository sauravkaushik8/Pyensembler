# pyensembler
  Allows to create millions of unique ensembles in a single line of code using Python.

  Takes any number of sklearn Machine Learning Models as Base Models and one Machine Learning Model as Top model to produce an ensemble of models.
  
                           Install from GitHub :  pip3 install git+git://github.com/sauravkaushik8/pyensembler.git
                                                           OR 
                                      Install from PyPi :  pip3 install pyensembler

### Example: 

```python
#Loading the package
from pyensembler import ensembler

#Loading models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#Loading dataset
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame(breast_cancer.data)
y = pd.DataFrame(breast_cancer.target)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3,random_state=0)

train = X_train
train['target'] = y_train

test = X_test
test['target'] = y_test

#Calling the ensembler function
stack = ensembler(train = train,
              test = test,
              base_model_list=[RandomForestClassifier(), GradientBoostingClassifier(), LogisticRegression()],
              nfold=3,
              seed=321,
              target_column='target',
              predictors = list(train.drop('target', axis =1).columns),
              top_model=LogisticRegression(),
              predict_prob = True)


#Training base models: RF, GBM, LR
stack.train_base_models()

#Training top model: LR
stack.train_top_model()

#Generating predictions
pred = stack.predictions()

#Generated predictions
pred[:5]
```
