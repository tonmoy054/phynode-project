#!/usr/bin/env python
# coding: utf-8

# In[298]:


from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
import glob
import sklearn


# <h2> Data Pre-Processing <h2>

# In[299]:


#importing all the csv files from directory

path =r'/home/tonmoy/Downloads/research project/Data_from_Abujar_Sir'
filenames = glob.glob(path + "/*.csv")


# In[300]:


dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))


# In[301]:


#concatanating all the files in a dataframe

df = pd.concat(dfs, axis=0, ignore_index=True)
df.columns


# In[302]:


#deleting columns

del df['Time,UNIX_T,RSSI1,RSSI2,RSSI3,Lux,Acc_x,Acc_y,Acc_z,Temp,ID,Pos,Si,Co,Ro']
del df['Co']
del df['Ro']
del df['Time']


# In[303]:


#dropping rows with Null values

df = df.dropna()


# In[304]:


df.columns


# In[305]:


x = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,11]].values
y = df.iloc[:,10].values


# In[306]:


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


# <h2> ML: Decision Tree  <h2>

# In[307]:


from sklearn.metrics import classification_report, accuracy_score


# In[308]:


from sklearn.tree import DecisionTreeClassifier
#creating the descision tree model

m = DecisionTreeClassifier(max_leaf_nodes=600)


# In[309]:


m.fit(x_train, y_train);
y_pred = m.predict(x_valid)
print(accuracy_score(y_valid, y_pred))


# In[310]:


def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 4)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


# In[311]:


m_rmse(m,x_valid, y_valid)


# <h2> ML: Random Forest And Feature Importance<h2>

# In[312]:


from sklearn.ensemble import RandomForestClassifier
m =RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state = 1,)


# In[313]:


m.fit(x_train, y_train);


# In[314]:


#function for feature importance dataframe
def feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[315]:


fi = m.feature_importances_
fi


# In[316]:


df1= pd.DataFrame({'Feature_names':df.iloc[:,[0,1,2,3,4,5,6,7,8,9,11]].columns, 'imp': fi})
df1.sort_values(by='imp', ascending=False)


# In[317]:


y_pred = m.predict(x_valid)
#print(accuracy_score(y_valid, y_pred))
y_pred.shape


# In[318]:


from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_valid, y_pred))
print(accuracy_score(y_valid, y_pred))


# In[319]:


m_rmse(m,x_valid, y_valid)


# <h2> Neural Network <h2>

# In[320]:


dep_var = 'Pos'


# In[321]:


df_nn = df
df_nn[dep_var] = np.log(df_nn[dep_var])


# In[322]:


df_nn.shape


# In[323]:


cont_nn,cat_nn = cont_cat_split(df_nn, dep_var=dep_var)


# In[324]:


procs_nn = []
to_nn = TabularPandas(df_nn, procs_nn, cat_nn, cont_nn,
                      splits=splits, y_names=dep_var)


# In[325]:


dls = to_nn.dataloaders(1024)


# In[326]:


y = to_nn.train.y
y.min(),y.max()


# In[327]:


learn = tabular_learner(dls, y_range=(4.70,5.70), layers=[100,300,500], metrics=[accuracy])


# In[328]:


learn.lr_find()


# In[329]:


learn.fit_one_cycle(25, 0.0003)


# In[330]:


preds,targs = learn.get_preds()
r_mse(preds,targs)


# In[ ]:




