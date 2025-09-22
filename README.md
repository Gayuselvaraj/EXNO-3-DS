# EXNO-3 - Feature Encoding and Transformation

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:

```python
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/ff8b04c7-5300-41e3-bdbe-942976c7535e)

```python
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/1b94dc48-3cb4-41e6-9549-2ac6922a8b9d)

```python
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/5a3eb6f2-30c2-4c33-b488-33a04cd2b927)

```python
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/8b3f2128-16f2-456f-8417-b4af620db05c)

```python
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```

```python
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/2831721a-83e8-4c11-b474-681b81f1dab2)

```python
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/75443d28-7237-41d3-aeb9-37acd15d362b)

```python
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

```python
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/e64d1912-62b5-4d10-ae13-ec92e09d66fd)

```python
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/92904e25-1092-4a26-b698-7e09bdf56a61)

```python
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/05890110-0a8a-43b7-ae84-63ef5789cbed)

```python
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/acd89ef9-2241-4440-a63d-729aa011eb3f)

```python
df.skew()
```
![image](https://github.com/user-attachments/assets/07595341-9095-491c-9179-b80139107db2)

```python
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/dd98af6f-843f-48bf-a55e-65b4a0fe3077)

```python
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/b57a0124-5401-44f7-bbe3-d8c00ab6db6e)


```python
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/41cf3184-4e0a-4d13-a665-4ed503133f3e)

```python
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/a937d30a-3b04-4dfb-af38-5e2f9a75b39c)

```python
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/a53df5e1-a00f-4d80-9dbc-374c47c4de66)

```python
df.skew()
```
![image](https://github.com/user-attachments/assets/d306ee12-f887-46d7-99f3-c31e314708b1)

```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/61a94e62-6a61-4964-97ad-e627a19ffcc9)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/2cc6afe1-be21-4b5d-97ff-9e8d59b3b2c4)

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b147373e-8d3b-4e43-a76c-e04c2132a4ac)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/fe55b4f3-d9c9-4424-a662-51f9eb5792de)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/18fb6edc-7beb-495a-9da4-da519dc24e81)

```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/58748b8d-09dc-45f1-a664-eb39f62c196d)

```python
dt=pd.read_csv("titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/b38de407-e41b-4c44-a2d4-292ab8c96e0e)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
![image](https://github.com/user-attachments/assets/17719962-20c0-4a97-a163-e7ed5231fcf7)

```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a27dbeb2-725d-43e0-8935-37d1fd4861ff)








## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

