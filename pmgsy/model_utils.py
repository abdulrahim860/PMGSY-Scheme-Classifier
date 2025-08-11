import joblib

def load_models():
    clf_pipeline = joblib.load("pkl/pmgsy_xgb_clf_pipeline.pkl")
    reg_pipeline = joblib.load("pkl/pmgsy_xgb_reg_pipeline.pkl")
    le_scheme = joblib.load("pkl/scheme_label_encoder.pkl")
    return clf_pipeline, reg_pipeline, le_scheme

def predict_scheme(clf_pipeline, le_scheme, data):
    preds_num = clf_pipeline.predict(data)
    preds = le_scheme.inverse_transform(preds_num)
    return preds

def predict_cost(reg_pipeline, data):
    preds = reg_pipeline.predict(data)
    return preds
