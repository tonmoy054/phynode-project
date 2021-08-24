#!/usr/bin/env python
# coding: utf-8

# In[217]:


from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from dtreeviz.trees import *
import glob
import sklearn


# <h2> Data Pre-Processing <h2>

# In[218]:


#importing all the csv files from directory

path =r'/home/tonmoy/Downloads/research project/Data_from_Abujar_Sir'
filenames = glob.glob(path + "/*.csv")


# In[219]:


dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))


# In[220]:


#concatanating all the files in a dataframe

df = pd.concat(dfs, axis=0, ignore_index=True)
df.columns


# In[221]:


#deleting columns

del df['Time,UNIX_T,RSSI1,RSSI2,RSSI3,Lux,Acc_x,Acc_y,Acc_z,Temp,ID,Pos,Si,Co,Ro']
del df['Co']
del df['Ro']
del df['Time']
#del df['UNIX_T']


# In[222]:


#dropping rows with Null values

df = df.dropna()


# In[223]:


df.columns


# In[224]:


#x = df.iloc[:,[0,1,2,3,4,5,6,7,8,10]].values
x = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,11]].values
y = df.iloc[:,10].values


# In[225]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split(x,y):
    return train_test_split(x,y,test_size = 0.2, random_state=0)

x_train, x_valid, y_train, y_valid = split(x,y)

def sc(x_train, x_valid): 
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_valid = sc.fit_transform(x_valid)
    return x_train, x_valid

x_train, x_valid = sc(x_train, x_valid)


# <h2> ML: Decision Tree And Feature Importance <h2>

# In[226]:


from sklearn.metrics import classification_report, accuracy_score


# In[227]:


from sklearn.tree import DecisionTreeClassifier
#creating the descision tree model

m = DecisionTreeClassifier(max_leaf_nodes=600)


# In[228]:


m.fit(x_train, y_train);
y_pred = m.predict(x_valid)
print(accuracy_score(y_valid, y_pred))


# In[229]:


def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 4)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


# In[230]:


m_rmse(m,x_valid, y_valid)


# <h2> ML: Random Forest <h2>

# In[231]:


from sklearn.ensemble import RandomForestClassifier
m =RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state = 1,)


# In[232]:


m.fit(x_train, y_train);


# In[233]:


#function for feature importance dataframe
def feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[234]:


fi = m.feature_importances_
fi


# In[235]:


df1= pd.DataFrame({'Feature_names':df.iloc[:,[0,1,2,3,4,5,6,7,8,9,11]].columns, 'imp': fi})
df1.sort_values(by='imp', ascending=False)


# In[236]:


y_pred = m.predict(x_valid)
#print(accuracy_score(y_valid, y_pred))
y_pred.shape


# In[237]:


from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_valid, y_pred))
print(accuracy_score(y_valid, y_pred))


# In[238]:


m_rmse(m,x_valid, y_valid)


# <h2> Neural Network <h2>

# In[239]:


dep_var = 'Pos'


# In[240]:


df_nn = df
df_nn[dep_var] = np.log(df_nn[dep_var])


# In[241]:


df_nn.shape


# In[242]:


cont_nn,cat_nn = cont_cat_split(df_nn, dep_var=dep_var)


# In[243]:


procs_nn = []
to_nn = TabularPandas(df_nn, procs_nn, cat_nn, cont_nn,
                      splits=splits, y_names=dep_var)


# In[244]:


dls = to_nn.dataloaders(1024)


# In[245]:


y = to_nn.train.y
y.min(),y.max()


# In[246]:


learn = tabular_learner(dls, y_range=(4.70,5.70), layers=[100,300,500], metrics=[accuracy])


# In[247]:


learn.lr_find()


# In[248]:


learn.fit_one_cycle(25, 0.0003)


# In[249]:


preds,targs = learn.get_preds()
r_mse(preds,targs)


# In[ ]:




