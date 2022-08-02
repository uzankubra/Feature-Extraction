#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


df= sns.load_dataset("titanic")


# In[4]:


df


# In[5]:


df.shape


# In[6]:


# Kadın ve erkek yolcuların sayısını bulalım
df["sex"].value_counts()


# In[7]:


# Kadın ve erkek yolcuların yuzdeligini bulmak ıstersek 
df["sex"].value_counts(normalize=True)


# In[8]:


# Her bir sütuna ait unique değerlerin sayısını bulunuz
df.nunique()


# In[9]:


# pclass degiskeninin unique değerlerinin sayisini bulunuz
df["pclass"].nunique()


# In[10]:


# pclass degiskeninin unique değerlerini bulunuz
df["pclass"].unique()


# In[11]:


# pclass ve parch degiskenlerinin unique degerlerini bulunuz 
df[["pclass","parch"]].nunique()


# In[12]:


# embarked degiskeninin tipini kontrol ediniz , tipini category olarak degistiriniz ve tekrar kontrol ediniz
df["embarked"].dtype
df["embarked"]=df["embarked"].astype("category")
df["embarked"].dtype


# In[13]:


# embarked degeri c olanların tum bılgılerını gosterınız
df[df["embarked"]=="C"]


# In[14]:


# yası 30dan kucuk ve kadın olan yolcuların tum bilgilerini gösteriniz
df[(df["sex"]=="female")&(df["age"]<30)]


# In[15]:


# fare 500 den buyuk veya yası 70den buyuk yolcularin bilgisini listeleyiniz
df[(df["fare"]>500)|(df["age"]>70)]


# In[16]:


# her bir degiskendeki bos degerlerin toplamını bulunuz
df.isnull().sum()


# In[17]:


# who degiskenini dataframeden düsürün
df= df.drop("who", axis=1)


# In[18]:


df


# In[19]:


# deck degiskenindeki bos degerleri deck degiskeninin en cok tekrar eden(mode) degeri ile doldurunuz
mod= df["deck"].mode()[0]


# In[20]:


mod


# In[21]:


df["deck"].fillna(mod, inplace=True)


# In[22]:


df.isnull().sum()


# In[23]:


# age degerindeki bos degerleri medyani ile doldurun
medyan=df["age"].median()
df["age"].fillna(medyan, inplace=True)


# In[24]:


df.isnull().sum()


# In[25]:


# survived degiskeninin pclass ve cinsiyet degiskenleri kiriliminda sum, count, mean degerlerini bulunuz 

# dataframe.groupby(kırılımlar).agg({ degiskenler, kolonlar: metrikler(sum, count, mean)})

df.groupby(["pclass","sex"]).agg({ "survived": ["count", "sum","mean"]})


# In[26]:


# 30 yasın altında olanlara bir otuza esit ve ustunde olanlara 0 verecek bir fonk yazınız 

# yazdıgınız fonksiyonu kullanarak titanik verisetinde age_flag adında bir degisken olusturunuz(apply ve lambda yapilari ile)

df["age_flag"]= df["age"].apply(lambda x: 1 if x<30 else 0) 


# In[27]:


df.head()


# In[28]:


# seaborn kütüphanesi icinden tips verisetini yukleyınız 

df= sns.load_dataset("tips")
df


# In[29]:


# time değişkeninin kategorilerine gore (launch, dinner) total_bill degerlerinin toplamını min max ve ortalamasını bulunuz.

df.groupby("time").agg({"total_bill": ["min", "max", "sum", "mean"]})


# In[30]:


# launch zamanına ve kadın müşterilere ait total bill ve tip değerlerinin daye göre toplamını, min, max ve ortalamasını bulunuz

df[(df["time"]=="Lunch")& (df["sex"]=="Female")].groupby("day").agg({"total_bill": ["min", "max", "sum", "mean"], "tip": ["min", "max", "sum", "mean"]})


# In[31]:


# size 3ten küçük total_bill'i 10dan büyük olan siparişlerin ortalaması nedir

df[(df["size"]<3)&(df["total_bill"]>10)]["total_bill"].mean()


# In[38]:


# total_bill degiskeninin kadin ve erkek icin ayri ayrı ortalamasını bulun

# buldugunuz ortalamanın altında olanlara 0, esit ve ya ustunde olanlara 1 verildigi yeni bir total_bill_flag olusturun

# Dikkat! Kadınların ortalaması ayrı, erkeklerin ortalaması ayrı degerlendirilecek

# parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak baslayın 

female_avg= df[df["sex"]=="Female"]["total_bill"].mean()

male_avg= df[df["sex"]== "Male"]["total_bill"].mean()

def function(sex, total_bill):
    if sex== "Female":
        if total_bill >= female_avg:
            return 1
        else:
            return 0
        
    else :
        if total_bill >= male_avg:
            return 1
        
        else:
            return 0
        
        
    
    


# In[41]:


df["total_bill_flag"]=df.apply(lambda x: function(x["sex"], x["total_bill"]), axis=1)


# In[42]:


df["total_bill_flag"]


# In[44]:


#total_bill_flag degiskenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz

df.groupby(["sex", "total_bill_flag"]).agg({"total_bill_flag": "count"})


# In[45]:


# total_bill_tip_sum degiskenine gore büyükten küçüğe sıralayınız ve ilk 30 kisiyi yeni bir dataframe'e atayınız

df["total_bill_tip_sum"]= df["total_bill"]+ df["tip"]


# In[47]:


df["total_bill_tip_sum"]


# In[48]:


first_30_df = df.sort_values("total_bill_tip_sum", ascending= False).head(30)


# In[49]:


first_30_df


# In[ ]:




