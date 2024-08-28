import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

st.markdown("""
            <center>
<h1 >EDA on Titanic DataSet <h1/>
            <center/>
""", unsafe_allow_html=True)

df=pd.read_csv("Titanic-Dataset.csv")

# Data Cleaning
df.drop("Cabin",axis=1,inplace=True)
df=df.dropna()
df["Survived"]=df["Survived"].replace(0, "not survived")
df["Survived"]=df["Survived"].replace(1, "survived")
df["Family size"]=df["SibSp"]+df["Parch"]
df.drop("Parch",axis=1,inplace=True)
df.drop("SibSp",axis=1,inplace=True)

options=None
li=["Data Overview", "Survival and Gender Distribution","Age and Survival Distribution","Pclass vs Other Distributions", "Fare vs other Distributions", "Embarked", "Summary of Insights"]
with st.sidebar:
    options=st.radio("", li)


if options==li[0] or options==None:
    st.title("Introduction")
    st.write("""
The Titanic dataset is a well-known dataset in the field of data science and machine learning. It contains information about passengers aboard the Titanic, including details such as their age, gender, class, fare, and whether they survived the disaster. This dataset is often used for predictive modeling and statistical analysis to understand factors that influenced survival rates. It's a popular choice for beginners in machine learning due to its accessibility and interesting historical context. 
""")
    
    st.title("Full Data of Titanic")
    st.write(df)

    st.title("Summary of the DataSet")
    st.write(df.describe())

    st.title("Co-relation matrix")
    dfCorr=df[["Pclass", "Age", "Fare", "Family size"]].corr()
    dfCorr

    st.title("Heat Map")
    fig, ax=plt.subplots(figsize=(10,8))
    sns.heatmap(dfCorr, annot=True)
    st.pyplot(fig)

if options==li[1]:
    st.title("Gender Distribution")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(df["Sex"], ax=ax)
    st.pyplot(fig)
    st.write("Here we can see that there were more males than females on the Titanic ship")
    "---"
    st.title("Survival Distribution")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(df["Survived"], ax=ax)
    st.pyplot(fig)
    st.write("As you can see from the above figure, a significant number of people did not survive the Titanic ship crash")
    "---"
    st.title("Survival vs Gender Distribution")
    fig, ax=plt.subplots(figsize=(10, 8))
    sns.histplot(df, x="Sex", hue="Survived", ax=ax, multiple='stack')
    st.pyplot(fig)
    st.write("Above, we observed that there were more males than females on the Titanic ship, but the survival rate of females was higher than that of males")

if options==li[2]:
    st.title("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 8))
    lineX1=[17, 17,17,17,17,17]
    lineX2=[37, 37,37,37,37,37]
    lineY=[100, 80,60,40,20,0]
    ax.plot(lineX1, lineY, color="g")
    ax.plot(lineX2, lineY, color="g")
    sns.histplot(df["Age"], ax=ax)
    st.pyplot(fig)
    st.write("Here, we can observe that most of the people on the ship were in the age range of 17-37.")
    "---"
   
    st.title("Survival vs Age Distribution")
    fig, ax=plt.subplots(figsize=(10, 8))
    lineX1=[17, 17,17,17,17,17]
    lineX2=[37, 37,37,37,37,37]
    lineY=[100, 80,60,40,20,0]
    ax.plot(lineX1, lineY, color="g")
    ax.plot(lineX2, lineY, color="g")
    sns.histplot(df, x='Age', hue='Survived', multiple='stack', bins=15,ax=ax)
    st.pyplot(fig)
    st.write("Above, we observed that there were more people in the age range of 20-40, and from the age of 17-47, the survival rate was low.")

    "---"
    st.title("Survival, Age and Gender Distribution")
    graph=sns.FacetGrid(df,row="Sex",col="Survived")
    graph.map(sns.histplot, "Age")
    st.pyplot(graph)
    st.write("In the age range of 20-30, most of the males did not survive the catastrophe.")
    st.write("In the age range of 20-40, most of the females survived the catastrophe.")



if options==li[3]:
    st.title("Pclass Distribution")
    fig, ax=plt.subplots(figsize=(10,8))
    sns.histplot(df["Pclass"], ax=ax)
    st.pyplot(fig)
    st.write("As we can see, there were more people in Pclass 3.")
    "---"

    st.title("Pclass vs Gender")
    fig, ax=plt.subplots(figsize=(10,8))
    sns.histplot(df,x="Pclass",hue="Sex",ax=ax, multiple="stack")
    st.pyplot(fig)
    st.write("As we can see, Pclass 1 had more females than the other two classes.")

    "---"

    st.title("Pclass vs Survied")
    fig, ax=plt.subplots(figsize=(10,8))
    sns.histplot(df,x="Pclass",hue="Survived",ax=ax, multiple="stack")
    st.pyplot(fig)
    st.write("As we can see, Pclass 1 had a higher survival rate than the other two classes.")
    "---"

    st.title("Pclass vs Age")
    fig, ax=plt.subplots(figsize=(10,8))
    sns.scatterplot(df, x="Pclass", y="Age", ax=ax)
    ax.plot()
    st.pyplot(fig)
    st.write("As we can see, the maximum age in Pclass 1 was 80, in Pclass 2 it was 70, and in Pclass 3 it was 70-75.")

    "---"

    st.title("Pclass , Males and Survival")
    fig, ax = plt.subplots(figsize=(10,8))
    pDf=df[df["Sex"] == "male"][["Pclass","Survived"]]
    sns.histplot(pDf, x="Pclass",hue="Survived", ax=ax, multiple="stack")
    st.pyplot(fig)
    st.write("As you can see, Pclass 1 had a higher male survival rate.")

    st.title("Pclass , Females and Survival")
    fig, ax = plt.subplots(figsize=(10,8))
    pDf=df[df["Sex"] == "female"][["Pclass","Survived"]]
    sns.histplot(pDf, x="Pclass",hue="Survived", ax=ax, multiple="stack")
    st.pyplot(fig)
    st.write("As you can see, Pclass 1 had a higher female survival rate.")

if options==li[4]:

    st.title("Fare vs Pclass")
    fig, ax=plt.subplots(figsize=(10,8))
    lineX=[1,2,3]
    lineY=[250,250,250]
    ax.plot(lineX, lineY, color="g")
    sns.scatterplot(df, x="Pclass", y="Fare", ax=ax)
    st.pyplot(fig)
    st.write("As we can see, only people in Pclass 1 had fares above the green line.")

    "---"
    st.title("Fare vs survival")
    fig, ax=plt.subplots(figsize=(10, 8))
    lineX=[90,90,90,90,90,90]
    lineY=[250, 200, 150, 100, 50,0]
    ax.plot(lineX,lineY,color="g")
    sns.histplot(df, x="Fare", hue="Survived", ax=ax, multiple="stack")
    st.pyplot(fig)
    st.write("As we can see, for fares greater than 90, the survival rate is high.")

    "---"
    st.title("Fare vs Sex")
    fig, ax=plt.subplots(figsize=(10, 8))
    lineX=[90,90,90,90,90,90]
    lineY=[250, 200, 150, 100, 50,0]
    ax.plot(lineX,lineY,color="g")
    sns.histplot(df, x="Fare", hue="Sex", ax=ax, multiple="stack")
    st.pyplot(fig)
    st.write("As we can see, for fares greater than 90, the number of females is high.")


    "---"
    st.title("Fare vs Age")
    fig, ax=plt.subplots(figsize=(10,8))
    lineX=[0, 10, 20, 30, 40, 50, 60, 70, 80]
    lineY=[250, 250, 250, 250, 250, 250, 250, 250, 250]
    ax.plot(lineX, lineY, color="g")
    sns.scatterplot(df, x="Age", y="Fare", ax=ax)
    st.pyplot(fig)

    st.write("As you can see from the chart, the green line represents fewer people, indicating that only a small number of them have a fare greater than half. Additionally, for the age range 15-70, the number of people with a fare greater than half is limited.")

    "---"

    st.title("Fare , Sex and Age")
    fig, ax=plt.subplots(figsize=(10,8))
    lineX=[0, 10, 20, 30, 40, 50, 60, 70, 80]
    lineY=[250, 250, 250, 250, 250, 250, 250, 250, 250]
    ax.plot(lineX, lineY, color="g")
    sns.scatterplot(df, x="Age", y="Fare", hue="Sex", ax=ax)
    st.pyplot(fig)

    st.write("As you can see, there were only four females and four males above the green line, and the maximum age of females was 64-66.")

    "---"

    st.title("Fare , Survival and Age")
    fig, ax=plt.subplots(figsize=(10,8))
    lineX=[0, 10, 20, 30, 40, 50, 60, 70, 80]
    lineY=[250, 250, 250, 250, 250, 250, 250, 250, 250]
    ax.plot(lineX, lineY, color="g")
    sns.scatterplot(df, x="Age", y="Fare", hue="Survived", ax=ax)
    st.pyplot(fig)

    st.write("Above the green line, we can see that 75% of the people are safe.")
    "---"
    st.title("Fare , Survival ,Males and Age")
    fig, ax=plt.subplots(figsize=(10,8))
    lineX=[0, 10, 20, 30, 40, 50, 60, 70, 80]
    lineY=[250, 250, 250, 250, 250, 250, 250, 250, 250]
    ax.plot(lineX, lineY, color="g")
    mDf=df[(df["Sex"]=="male")][["Fare", "Age", "Survived"]]
    # mDf
    sns.scatterplot(mDf, x="Age", y="Fare", hue="Survived", ax=ax)
    st.pyplot(fig)

    st.write("Above the green line, we can see that 50% of males are safe.")

    st.title("Fare , Survival ,Females and Age")
    fig, ax=plt.subplots(figsize=(10,8))
    lineX=[0, 10, 20, 30, 40, 50, 60, 70, 80]
    lineY=[250, 250, 250, 250, 250, 250, 250, 250, 250]
    ax.plot(lineX, lineY, color="g")
    mDf=df[(df["Sex"]=="female")][["Fare", "Age", "Survived"]]
    # mDf
    sns.scatterplot(mDf, x="Age", y="Fare", hue="Survived", ax=ax)
    st.pyplot(fig)

    st.write("Above the green line, we can see that 100% of females are safe.")


if options==li[5]:
    # df["Embarked"].value_counts()
    fig, ax=plt.subplots(figsize=(10, 8))
    al=st.multiselect("slect the option", ["S","C","Q"])
    emblist=[0, 0, 0]
    emblist[0]+=0.2 if 'S' in al else 0
    emblist[1]+=0.2 if 'C' in al else 0
    emblist[2]+=0.2 if 'Q' in al else 0
    ax.pie(x=[df["Embarked"].value_counts()["S"], df["Embarked"].value_counts()["C"], df["Embarked"].value_counts()["Q"]],labels=df["Embarked"].unique(),rotatelabels=True, explode=emblist)
    st.pyplot(fig)
    st.write("As you can see, more people embarked from S.")


if options==li[-1]:
    st.write("""
1) Gender Distribution:
The Titanic had more males than females.
             

1) Survival Distribution:
A significant number of people did not survive the Titanic crash.
             

1) Survival vs Gender Distribution:
Although there were more males on the Titanic, the survival rate for females was higher.
             

1) Age Distribution:
Most people on the ship were in the age range of 17-37.
             
1) Survival vs Age Distribution:
The survival rate was low for people aged 20-40.
             
1) Pclass Distribution:
Pclass 3 had the highest number of people.
             
1) Pclass vs Gender:
Pclass 1 had more females than the other two classes.
             
1) Pclass vs Survived:
Pclass 1 had a higher survival rate than the other two classes.
             
1) Fare vs Pclass:
Only people in Pclass 1 had fares above a certain threshold.
             
1) Fare vs Survival:
The survival rate was high for fares greater than 90.
             
1) Embarked:
More people embarked from S than from C or Q.
             
""")