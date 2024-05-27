import pickle
import json
import sklearn
import joblib
def savefile(regression_model):
    with open('model_pickel','wb') as f:
        pickle.dump(regression_model,f)
    with open('model_pickel','rb') as f:
        rm =pickle.load(f)
    input_area = float(input("Enter the area from savefile: "))

    # Predict the price based on the linear regression model
    predicted_price =  rm.predict([[input_area]])

    # Print the predicted price
    print(f"Predicted Price form  the save file  {input_area} square feet: {predicted_price[0]} INR")

def joblibsave(regression_model):
    joblib.dump(regression_model,'regression_model_joblib')
    rmj=joblib.load('regression_model_joblib')
    input_area = float(input("Enter the area from joblibsave: "))

    # Predict the price based on the linear regression model
    predicted_price = rmj.predict([[input_area]])

    print(f"Predicted Price form  the joblibsave  {input_area} square feet: {predicted_price[0]} INR")