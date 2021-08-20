# modelop.schema.0: input_schema.avsc
# modelop.schema.1: output_schema.avsc

import pandas as pd
import pickle
import numpy as np
import Algorithmia
import os
from pandas.core.frame import DataFrame
from datetime import datetime

# Bias libraries
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias 

MY_API_KEY = simFGPugHNKuIMaw+23od1cnGjC1 #os.getenv("ALGO_API_KEY_POC", None) #simFGPugHNKuIMaw+23od1cnGjC1
ALGORITHMIA_API = https://api.bnymellon.productionize.ai # os.getenv('ALGO_API_ADDRESS_POC', None) #https://api.bnymellon.productionize.ai
output_file_name = ""
output_file_path = ""
decisions = []

client = Algorithmia.client(MY_API_KEY, ALGORITHMIA_API)
algo = client.algo('bny_poc/german_credit/0.1.0')
algo.set_options(timeout=3000)  # optional

# API calls will begin at the apply() method, with the request body passed as 'input'
# For more details, see algorithmia.com/developers/algorithm-development/languages



def apply(input):
    batch_input = pd.read_csv(
        client.file('data://bny_poc/German_Credit/X_train.json').getFile().name
    )
    print(batch_input.shape)
    records = batch_input.to_dict("records")


    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_file_name = "bny_poc_german_credit_" + \
        str(timestamp) + ".csv"
    print(str(output_file_name))


    output_dictory_path = "data://bny_poc/german_credit_output/"
    output_file_path = output_dictory_path + output_file_name
    print(str(output_file_path))


    batch_input = pd.read_csv(
        client.file(input).getFile().name
    )


    print(batch_input.shape)


    records = batch_input.to_dict("records")


    count = 0


    for record in records:
        result = algo.pipe(record).result
        timestamp = {"timestamp": datetime.now().strftime(
            "%Y-%m-%d_%H:%M:%S.%f")[:-3]}
        results = dict(record, **result, **timestamp)
        print(str(results))
        decisions.append(results)
        DataFrame.from_dict(decisions).to_csv(output_file_name)
        client.file(output_file_path).putFile(output_file_name)
        count += 1
        print(str(count))


        if count > 30000:
            break


    return f"{count} records processed, result stored at: {output_file_path}"


# modelop.init
def begin():
    
    global logreg_classifier
    
    # load pickled logistic regression model
    logreg_classifier = pickle.load(open("logreg_classifier.pickle", "rb"))

    
# modelop.score
def action(data):
    
    # Turn data into DataFrame
    data = pd.DataFrame([data])
    
    # There are only two unique values in data.number_people_liable.
    # Treat it as a categorical feature
    data.number_people_liable = data.number_people_liable.astype('object')

    predictive_features = [
        'duration_months', 'credit_amount', 'installment_rate',
        'present_residence_since', 'age_years', 'number_existing_credits',
        'checking_status', 'credit_history', 'purpose', 'savings_account',
        'present_employment_since', 'debtors_guarantors', 'property',
        'installment_plans', 'housing', 'job', 'number_people_liable',
        'telephone', 'foreign_worker'
    ]
    
    data["predicted_score"] = logreg_classifier.predict(data[predictive_features])
    
    # MOC expects the action function to be a *yield* function
    yield data.to_dict(orient="records")


# modelop.metrics
def metrics(data):
    
    data = pd.DataFrame(data)

    # To measure Bias towards gender, filter DataFrame
    # to "score", "label_value" (ground truth), and
    # "gender" (protected attribute)
    data_scored = data[["score", "label_value", "gender"]]

    # Process DataFrame
    data_scored_processed, _ = preprocess_input_df(data_scored)

    # Group Metrics
    g = Group()
    xtab, _ = g.get_crosstabs(data_scored_processed)

    # Absolute metrics, such as 'tpr', 'tnr','precision', etc.
    absolute_metrics = g.list_absolute_metrics(xtab)

    # DataFrame of calculated absolute metrics for each sample population group
    absolute_metrics_df = xtab[
        ['attribute_name', 'attribute_value'] + absolute_metrics].round(2)

    # For example:
    """
        attribute_name  attribute_value     tpr     tnr  ... precision
    0   gender          female              0.60    0.88 ... 0.75
    1   gender          male                0.49    0.90 ... 0.64
    """

    # Bias Metrics
    b = Bias()

    # Disparities calculated in relation gender for "male" and "female"
    bias_df = b.get_disparity_predefined_groups(
        xtab,
        original_df=data_scored_processed,
        ref_groups_dict={'gender': 'male'},
        alpha=0.05, mask_significance=True
    )

    # Disparity metrics added to bias DataFrame
    calculated_disparities = b.list_disparities(bias_df)

    disparity_metrics_df = bias_df[
        ['attribute_name', 'attribute_value'] + calculated_disparities]

    # For example:
    """
        attribute_name	attribute_value    ppr_disparity   precision_disparity
    0   gender          female             0.714286        1.41791
    1   gender          male               1.000000        1.000000
    """

    output_metrics_df = disparity_metrics_df # or absolute_metrics_df

    # Output a JSON object of calculated metrics
    yield output_metrics_df.to_dict(orient="records")
