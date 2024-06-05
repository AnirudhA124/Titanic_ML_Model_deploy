from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('titanic_model_logistic_regression2.pkl','rb'))
ohe=pickle.load(open('OneHotEncoder.pkl','rb'))
scaler=pickle.load(open('StandarScaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pclass = float(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = float(request.form['sibsp'])
        parch = float(request.form['parch'])
        fare = float(request.form['fare'])
        sex = request.form['sex']
        embarked = request.form['embarked']
        ohe_input=[[sex,embarked]]
        print(ohe_input)
        ohe_transformed=ohe.transform(ohe_input)
        ohe_transformed_array = ohe_transformed.toarray()
        print(ohe_transformed_array)
        input=np.array([pclass,age,sibsp,parch,fare],dtype=object).reshape(1,5)
        input_output=scaler.transform(input)
        print(input_output)
        print(input)
        input_final=np.hstack((input_output,ohe_transformed_array))
        print(input_final)
        # Process the input data here (e.g., make predictions with a machine learning model)
        
        # For demonstration purposes, simply printing the input data
        print("Pclass:", pclass)
        print("Age:", age)
        print("SibSp:", sibsp)
        print("Parch:", parch)
        print("Fare:", fare)
        print("Sex:", sex)
        print("Embarked:", embarked)
        result=model.predict(input_final)
        if result[0]==0:
            message="Did not survived!"
            print("Dead")
        else:
            message="Survived!"
            print("Survived")
        return render_template('index.html',message=message)

if __name__ == '__main__':
    app.run(debug=True)
