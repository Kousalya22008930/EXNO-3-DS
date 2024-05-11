## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/d876dc9b-1caf-4b06-8ecd-bcda1c6000af)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/0c2bd281-3bdf-4e66-9f0a-76be8f5da874)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/bd47a779-b6fa-4151-b94a-d7340d7b4e2a)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/91330843-75a0-4046-9e5d-a322160aa03b)

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/46c23a23-fb39-445f-a1af-cf1871afe6f0)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/be7a781f-e3b2-4ebf-93b4-066a0f7ff742)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/2efc44c1-6731-433a-a05c-258848a4acdb)

```
pip install --upgrade category_encoders
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/044c0252-0270-4612-9fb7-01a53eaf9fa8)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/9c503bdc-c997-4d88-9307-58bd693bb070)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/0470d70d-73b8-436b-89f0-e18736d869fe)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/c4b94a39-e0b3-49fb-90f0-8f8a48c71d1a)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/8cc0ad5e-3973-4e7f-8b0e-1f2fff8610b6)

```
df.skew()
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/96f036f1-956d-4f44-aee9-fc9b415b21e2)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/24d803cf-ddb3-4bfd-a730-e9941df9f085)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/542ade73-c4ec-4e17-a24f-9928fa7c36b4)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/e31c4f3b-ebbf-4ef1-9c11-46e9a8ce1459)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/28032b96-8925-470d-9136-e69ffdf8d279)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/a7f32738-c5e2-4713-b63a-1ea796b5a00f)

```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/8bc631ea-30be-4a9a-af9e-205e3f0e4fa5)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/43e6d685-77e4-4e2c-aa45-132043b0a2d5)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/715824d9-d235-4d8c-9613-bbf7eceb16d7)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/98aa3c09-7c18-447f-8f67-193dd5a12b22)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/9f265833-6cf9-469c-adf0-31881f2b0e7a)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/e619725a-ec8d-4b57-b0e7-dd9220fcc1b4)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/Kousalya22008930/EXNO-3-DS/assets/119389108/fff9bde9-37de-409d-875d-348618d1a403)



# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
