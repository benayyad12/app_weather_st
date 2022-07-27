import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np



st.write('''
   # Weather Prediction
''')


from model import predict
classes = {0: 'rain', 1: 'snow', 2: 'sun', 3: 'drizzle', 4: 'fog'}
class_labels = list(classes.values())
st.markdown('**Objective** : Given details about the weather we are trying to predict.')
st.markdown('The model can predict if it belongs to the following 5 categories : ***  rain *** snow *** sun *** drizzle *** fog')
def predict_class():
    data = list(map(float,[precipitation,temp_max,temp_min, wind]))
    result, probs = predict(data)
    st.write("The predicted class is ",result)
    probs = [np.round(x,6) for x in probs]
    ax = sns.barplot(probs ,class_labels, palette="winter", orient='h')
    ax.set_yticklabels(class_labels,rotation=0)
    plt.title("Probabilities of the Data belonging to each class")
    for index, value in enumerate(probs):
        plt.text(value, index,str(value))
    st.pyplot()
st.markdown("**Please enter the parameters of weather we are going to predict **")
precipitation=st.sidebar.slider('precipitation',0,55.9)
temp_max=st.sidebar.slider('temp_max',-1.6,35.6)
temp_min=st.sidebar.slider('temp_min',-7.1,18.3)
wind=st.sidebar.slider('wind',0.4,9.5)
if st.button("Predict"):
    predict_class()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
st.set_option('deprecation.showPyplotGlobalUse', False)
