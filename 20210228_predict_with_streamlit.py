# https://docs.streamlit.io/en/stable/api.html#streamlit.slider
import streamlit as st
import pandas

# load model
from joblib import load
rf = load('model2.joblib')

def convert_age(age1):
    bins = [0, 12, 18, 35, 50, 80]
    for index1, val1 in enumerate(bins[1:]):
        if age1 < val1:
            return index1
    return 4

scaler = load('scaler.joblib')
def convert_fare(fare1):
    return scaler.transform([[fare1]])

dict1 = { 'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}
def convert_embark_town(embark1):
    x = dict1[embark1]
    if x==0:
        return 1,0,0
    elif x==1:
        return 0,1,0
    else:
        return 0,0,1
        
def convert_sex(sex1):
    return 1 if sex1 == '男性' else 0



# 畫面設計
st.markdown('# 生存預測系統')
pclass_series = pandas.Series([1, 2, 3])
sex_series = pandas.Series(['男性', '女性'])
embark_town_series = pandas.Series(['Cherbourg', 'Queenstown', 'Southampton'])

sex = st.sidebar.selectbox('性別:', sex_series)
# '性別:', sex

age = st.sidebar.slider('年齡', 0, 100, 20)
# '年齡:', age

sibsp = st.sidebar.slider('兄弟姊妹同行人數', 0, 10, 0)
# '兄弟姊妹同行人數:', sibsp

parch = st.sidebar.slider('父母子女同行人數', 0, 10, 0)
# '父母子女同行人數:', parch

embark_town = st.sidebar.selectbox('上船港口:', embark_town_series)
# '上船港口:', embark_town

pclass = st.sidebar.selectbox('艙等:', pclass_series)
# '艙等:', pclass

fare = st.sidebar.slider('票價', 0, 100, 20)
# '票價:', fare


if st.sidebar.button('預測'):
    '性別:', sex
    '年齡:', age
    '兄弟姊妹同行人數:', sibsp
    '父母子女同行人數:', parch
    '上船港口:', embark_town
    '艙等:', pclass
    '票價:', fare

    # predict
    X = []
    # pclass	sex	age	sibsp	parch	fare    adult_male	embark_town
    
    adult_male = 1 if age>18 and sex == 1 else 0
    X.append([pclass, convert_sex(sex), convert_age(age), sibsp, parch, convert_fare(fare), adult_male, *convert_embark_town(embark_town)])

    if rf.predict(X) == 1:
        st.markdown('==> **生存**')
    else:
        st.markdown('==> **死亡**')
        
        
    st.markdown(f'生存機率={rf.predict_proba(X)[0][1] * 100:.2f}%')

