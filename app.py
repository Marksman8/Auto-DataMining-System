from flask import Flask, request, render_template
import pandas as pd
from modules.preprocessing import preprocess_data
from modules.visualize import generate_visuals
from modules.association import run_association_mining
from modules.clustering import run_clustering
from modules.automl import run_automl

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)

        df_clean = preprocess_data(df)
        df_clean.to_csv("static/cleaned_dataset.csv", index=False)
        visuals = generate_visuals(df_clean)
        print("Any NaNs in cleaned data:", df_clean.isnull().sum().sum())  # should be 0
        assoc_rules = run_association_mining(df_clean)
        clusters = run_clustering(df_clean)
        model_result = run_automl(df_clean)

        return render_template("report.html", visuals=visuals, rules=assoc_rules, clusters=clusters, model=model_result)
    return render_template("upload.html")

  # should be 0


if __name__ == '__main__':
    app.run(debug=True)
