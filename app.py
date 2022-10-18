from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import joblib
app = Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("main.html")


@app.route('/predecir',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    predictor = joblib.load('modelo_xgboost.pkl')
    predicccion = predictor.predict(final);
    return render_template('main.html',pred='Las ventas del producto en la tienda con los atributos ingresadas es  {}'.format(predicccion))



if __name__ == '__main__':
    app.run(debug=True)
