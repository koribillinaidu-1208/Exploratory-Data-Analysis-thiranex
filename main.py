import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.rcParams["figure.figsize"]=(14,10)

df=pd.read_csv("train.csv")

df.drop_duplicates(inplace=True)

df["Age"].fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)

df.drop(["Cabin","Name","Ticket","PassengerId"],axis=1,inplace=True)

df["Sex"]=df["Sex"].map({"male":0,"female":1})
df["Embarked"]=df["Embarked"].map({"S":0,"C":1,"Q":2})

fig,axs=plt.subplots(3,3)

sns.countplot(x=df["Survived"],ax=axs[0,0])
sns.countplot(x=df["Sex"],ax=axs[0,1])
df["Age"].hist(bins=30,ax=axs[0,2])

sns.countplot(x="Survived",hue="Sex",data=df,ax=axs[1,0])
sns.countplot(x="Survived",hue="Pclass",data=df,ax=axs[1,1])
sns.boxplot(x="Survived",y="Age",data=df,ax=axs[1,2])

corr=df.select_dtypes(include=["number"]).corr()
sns.heatmap(corr,annot=True,cmap="coolwarm",ax=axs[2,0])

axs[2,1].axis("off")
axs[2,2].axis("off")

plt.tight_layout()
plt.show()

df.to_csv("cleaned_titanic.csv",index=False)
