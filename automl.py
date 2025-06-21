from pycaret.classification import *

def run_automl(df):
    target = df.columns[-1]  # last column is assumed to be the target
    setup(df, target=target, verbose=False, session_id=123)
    best_model = compare_models()
    return str(best_model)
