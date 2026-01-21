My first project.

https://www.kaggle.com/competitions/titanic

It seems like the 'hello world' in ML.

## Data Processing:

After we get the dataset, we need to process the data first:

``` python
print(train.isnull().sum())
print(test.isnull().sum())
```
This code is used to count the number of missing values, and we will know number of missing values in each feature.

After we know the number of missing values, we can process the data by using these method:

There are 327 missing values in Cabin and we are hard to fill it, so we can just delete this feature.

``` python
train = train.drop('Cabin', axis = 1)
test = test.drop('Cabin', axis = 1)
```

We can use the median to fill in the missing values of age. I believe it is better than the mean because age has a skewed distribution in this dataset.



``` python
age_median = train['Age'].median()
train['Age'] = train['Age'].fillna(age_median)
test['Age'] = test['Age'].fillna(age_median)
```
