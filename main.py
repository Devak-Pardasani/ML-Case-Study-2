from preprocessing import load_and_preprocess
from model_nn import train_nn
from model_xgb import train_xgb
from lstm_model import train_lstm
from train_res import train_resnet_fc
from train_light_gbm import train_lgb
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess(raw=False)
    #model = train_resnet_fc(X_train, X_test, y_train, y_test)
    model = train_xgb( X_train, X_test, y_train, y_test)
    #model = train_xgb(X_train, X_test, y_train, y_test)
    #model = train_lstm(X_train,X_test,y_train,y_test);
    #model2 = train_nn(X_train, X_test, y_train, y_test)