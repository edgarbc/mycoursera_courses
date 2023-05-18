export REGION="europe-west4"
export PROJECT_ID=$(gcloud config list --format 'value(core.project)')
export BUCKET_NAME=$PROJECT_ID"-bucket"

gsutil mb -l $REGION "gs://"$BUCKET_NAME

cd /home/jupyter/titanic
pip install setuptools
python setup.py install


// Note: You can ignore the error: google-auth 2.3.3 is installed but google-auth<2.0dev,>=1.25.0 is required by {'google-api-core'}, 
// as it does not affect the lab functionality

// verify that this runs (training, test, and validation datasets are the same for testing purposes)
python -m trainer.task -v \
    --model_param_kernel=linear \
    --model_dir="gs://"$BUCKET_NAME"/titanic/trial" \
    --data_format=bigquery \
    --training_data_uri="bq://"$PROJECT_ID".titanic.survivors" \
    --test_data_uri="bq://"$PROJECT_ID".titanic.survivors" \
    --validation_data_uri="bq://"$PROJECT_ID".titanic.survivors"
