#!/usr/bin/env python
# coding: utf-8

# <h2> Importing Libraries <h2>

# In[939]:


import fastbook
fastbook.setup_book()


# In[940]:


from fastbook import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from dtreeviz.trees import *
import glob
import sklearn


# <h2> Data Pre-Processing <h2>

# In[941]:


#importing all the csv files from directory

path =r'/home/tonmoy/Downloads/research project/Data_from_Abujar_Sir'
filenames = glob.glob(path + "/*.csv")


# In[942]:


dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))


# In[943]:


#concatanating all the files in a dataframe

df = pd.concat(dfs, axis=0, ignore_index=True)


# In[944]:


df.columns


# In[945]:


#deleting columns

del df['Time,UNIX_T,RSSI1,RSSI2,RSSI3,Lux,Acc_x,Acc_y,Acc_z,Temp,ID,Pos,Si,Co,Ro']
del df['Co']
del df['Ro']
del df['Time']
del df['UNIX_T']


# In[946]:


df.columns


# In[947]:


#dropping rows with Null values

df = df.dropna()


# In[948]:


#declaring dependable varriable

dep_var = 'Pos'
df[dep_var] = np.log(df[dep_var])


# In[949]:


procs = []


# In[950]:


#splitting train and validation dataset 80-20 ratio

train_idx = range(0, Int(len(df)*.8)-1)
valid_idx = range(Int(len(df)*.8), len(df))
splits = (list(train_idx),list(valid_idx))


# In[951]:


#splitting continuous and categorical varriable

cont,cat = cont_cat_split(df, dep_var=dep_var)


# In[952]:


#creating tabular object

to = TabularPandas(df, procs,cont,cat, y_names=dep_var, splits=splits)


# In[953]:


#length of training and validation set

len(to.train),len(to.valid)


# In[954]:


#splitting all the train set and validation set
#y is the target

xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y


# <h2> ML: Decision Tree And Feature Importance <h2>

# In[955]:


from sklearn.tree import DecisionTreeRegressor
#creating the descision tree model

m = DecisionTreeRegressor(max_leaf_nodes=10)


# In[956]:


#function for feature importance dataframe
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[957]:


m.fit(xs,y)


# In[958]:


fi = rf_feat_importance(m, xs)
fi


# In[959]:


#plotting

fi.plot('cols', 'imp', 'barh', figsize=(12,7))


# In[960]:


to_keep = fi[fi.imp>0.005].cols


# In[961]:


xs = xs[to_keep]
valid_xs = valid_xs[to_keep]


# In[962]:


m.fit(xs,y)
fi = rf_feat_importance(m, xs)
fi


# In[963]:


#sample draw with 10 nodes in the tree

draw_tree(m, xs, size=10, leaves_parallel=True, precision=2)


# In[964]:


def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 4)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


# In[965]:


#Training on training set

m_rmse(m, xs, y)


# In[966]:


#validation set prediction

m_rmse(m, valid_xs, valid_y)


# In[967]:


m.get_n_leaves(), len(xs)


# In[968]:


#another tree with min sample leaf 35

m = DecisionTreeRegressor(min_samples_leaf=35)
m.fit(xs, y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# In[969]:


m.score(valid_xs, valid_y)


# <h2> ML: Random Forest <h2>

# In[970]:


#n_estimators defines the number of trees
#max_samples defines how many rows to sample for training each tree
#max_features defines how many columns
from sklearn.ensemble import RandomForestRegressor

def rf(xs, y, n_estimators=40, max_samples=20000,
       max_features=0.5, min_samples_leaf=5):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf , oob_score=True).fit(xs, y)


# In[971]:


m = rf(xs, y);


# In[972]:


m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)


# In[973]:


m.score(valid_xs, valid_y)


# <h2> Neural Network <h2>

# In[974]:


#dataframe and dep var for NN

df_nn = df
df_nn[dep_var] = np.log(df_nn[dep_var])


# In[975]:


df_nn.shape


# In[976]:


#Splitting cont and cat var for NN

cont_nn,cat_nn = cont_cat_split(df_nn, dep_var=dep_var)


# In[977]:


#Tabular object for NN

procs_nn = []
to_nn = TabularPandas(df_nn, procs_nn, cat_nn, cont_nn,
                      splits=splits, y_names=dep_var)


# In[978]:


#dataloader with a batch size of 1024

dls = to_nn.dataloaders(1024)


# In[979]:


# min and max of y for better pred

y = to_nn.train.y
y.min(),y.max()


# In[980]:


#learner model

learn = tabular_learner(dls, y_range=(1.55,1.74), loss_func=F.mse_loss)


# In[981]:


#10 epochs with 1e^-2 learning rate

learn.fit_one_cycle(20, 1e-2)


# In[982]:


preds,targs = learn.get_preds()
r_mse(preds,targs)

