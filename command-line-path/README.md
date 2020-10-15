# Amazon Comprehend Custom Classification

This is the command line version for creating a Amazon Comprehend Custom Classifier model.

* [Overview](#overview)
* [Command line prerequisites](#command-line-prerequisites)
* [Detailed Steps](#detailed-steps)
  * [Creating the bucket](#creating-the-bucket)
  * [Configure an IAM role](#configure-an-iam-role)
  * [Preparing the data](#preparing-the-data)
  * [Uploading the data](#uploading-the-data)
  * [Training the model](#training-the-model)
  * [Inference](#inference)

## Overview

The custom classifier workload is built in two steps:

1. Training the custom model – no particular machine learning or deep learning knowledge is necessary
2. Classifying new data

Steps to follow are relatively simple:

1. Create a bucket that will host training data
2. Create a bucket that will host training data artifacts and production results. That can be the same
3. Configure an IAM role allowing Comprehend to [access newly created buckets](https://docs.aws.amazon.com/comprehend/latest/dg/access-control-managing-permissions.html#auth-role-permissions)
4. Prepare data for training
5. Upload training data in the S3 bucket
6. Launch a “Train Classifier” job from the console: “Amazon Comprehend” > “Custom Classification” > “Train Classifier”
7. Prepare data for classification (one text per line, no header, same format as training data). Some more details [here](https://docs.aws.amazon.com/comprehend/latest/dg/how-class-run.html)
8. Launch a custom classification job
9. Gather results: a file name output.tar.gz is generated in the destination bucket. File format is [JSON Line]( https://docs.aws.amazon.com/comprehend/latest/dg/how-class-run.html).

## Command line prerequisites

You have anaconda [available](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html).

Create the conda environment for data preparation:

```shell
$> conda create --name comprehendCustomClassification python=3.7 pandas tqdm ipython
```

Activate conda environment:

```shell
$> conda activate comprehendCustomClassification
```

## Detailed Steps

Now, it is time to get our hands dirty.

### Creating the bucket

The following command creates the bucket `hervenivon-poc`. As bucket names are unique, please change it to your desire.

```shell
$> aws s3api create-bucket --acl private --bucket `hervenivon-poc` --region us-east-1
```

You should see something like:

```json
{
    "Location": "/hervenivon-poc"
}
```

Note 💡: if you want to create your bucket in another location you must add a Location Constraint. Example:

```shell
$> aws s3api create-bucket --bucket my-bucket --region eu-west-1 --create-bucket-configuration LocationConstraint=eu-west-1
```

### Configure an IAM role

In order to authorize Amazon Comprehend to perform bucket reads and writes during the training or during the inference, we must grant Amazon Comprehend access to the Amazon S3 bucket that we created.

We are going to create a data access role in our account to trust the Amazon Comprehend service principal.

Create a file `ComprehendBucketAccessRole-TrustPolicy.json` that contains the role’s trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "comprehend.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Create a file `ComprehendBucketAccessRole-Permissions.json` that contains the following access policy. Please change bucket name accordingly to the bucket you created.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::hervenivon-poc/*"
            ],
            "Effect": "Allow"
        },
        {
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::hervenivon-poc"
            ],
            "Effect": "Allow"
        }
    ]
}
```

The following command create the role:

```shell
$> aws iam create-role --role-name ComprehendBucketAccessRole --assume-role-policy-document file://ComprehendBucketAccessRole-TrustPolicy.json
```

You should see something like:

```shell
{
    "Role": {
        "Path": "/",
        "RoleName": "ComprehendBucketAccessRole",
        "RoleId": "AROAUS7UWFDI7L3MYSW7B",
        "Arn": "arn:aws:iam::312306070809:role/ComprehendBucketAccessRole",
        "CreateDate": "2019-06-27T09:02:50Z",
        "AssumeRolePolicyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "comprehend.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
    }
}
```

Now we must attach permissions to the role:

```shell
$> aws iam put-role-policy --role-name ComprehendBucketAccessRole --policy-name BucketAccessPolicy --policy-document file://ComprehendBucketAccessRole-Permissions.json
```

You should see no output.

### Preparing the data

Once you have downloaded the data from the mentioned Drive, you get a zip file `yahoo_answers_csv.tar.gz` containing four files:

- `classes.txt`
- `readme.txt`
- `test.csv`
- `train.csv`

As per the readme:

_The files train.csv and test.csv contain all the training samples as comma-separated values. There are 4 columns in them, corresponding to class index (1 to 10), question title, question content and best answer. The text fields are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). New lines are escaped by a backslash followed with an "n" character, that is "\n"._

Overview of file content

```csv
"5","why doesn't an optical mouse work on a glass table?","or even on some surfaces?","Optical mice use an LED
"6","What is the best off-road motorcycle trail ?","long-distance trail throughout CA","i hear that the mojave
"3","What is Trans Fat? How to reduce that?","I heard that tras fat is bad for the body.  Why is that? Where ca
"7","How many planes Fedex has?","I heard that it is the largest airline in the world","according to the www.fe
"7","In the san francisco bay area, does it make sense to rent or buy ?","the prices of rent and the price of b
```

The file `classes.txt` contains the label for each line:

1. Society & Culture
2. Science & Mathematics
3. Health
4. Education & Reference
5. Computers & Internet
6. Sports
7. Business & Finance
8. Entertainment & Music
9. Family & Relationships
10. Politics & Government

`train.csv` contains 1400000 lines and `test.csv` 60000 lines. Amazon Comprehend uses between 10 and 20 percent of the documents submitted for training to test the custom classifier.

The following command indicates us that the data are evenly distributed.

```shel
$> awk -F '","' '{print $1}'  yahoo_answers_csv/train.csv | sort | uniq -c
```

140000 lines per label. Amazon Comprehend “recommend[s] that you train the model with up to 1,000 training documents for each label.” and no more than 1000000 documents.

Amazon Comprehend recommends the following:

> For each class, provide a minimum of 10 documents for training. For example, if you have 10 possible classes, you need a total of at least 100 classified documents to train the model. For more accurate training, we recommend at least 50 documents or more for each class. While a minimum of 10 training documents for each class is required, you get better accuracy with more documents. The total size of the training documents must be less than 5 GB.

With 20 percent of 1000000 use for test, that is still plenty of data to train our custom classifier.

So, we are going to use a shortened version of `train.csv` to train our custom comprehend model and we are going to use `test.csv` to perform our validation and see how well our custom model performs.

For training, the file format must conform with the [following](https://docs.aws.amazon.com/comprehend/latest/dg/how-document-classification-training.html):

- File must contain one label and one text per line – 2 columns
- No header
- Format UTF-8, carriage return “\n”.

Labels “must be uppercase, can be multitoken, have whitespace, consist of multiple words connect by underscores or hyphens or may even contain a comma in it, as long as it is correctly escaped.”

Here are the proposed labels:

| Index | Original | For training |
| --- | --- | --- |
| 1 | Society & Culture | SOCIETY_AND_CULTURE |
| 2 | Science & Mathematics | SCIENCE_AND_MATHEMATICS |
| 3 | Health | HEALTH |
| 4 | Education & Reference | EDUCATION_AND_REFERENCE |
| 5 | Computers & Internet | COMPUTERS_AND_INTERNET |
| 6 | Sports | SPORTS |
| 7 | Business & Finance | BUSINESS_AND_FINANCE |
| 8 | Entertainment & Music | ENTERTAINMENT_AND_MUSIC |
| 9 | Family & Relationships | FAMILY_AND_RELATIONSHIPS |
| 10 | Politics & Government | POLITICS_AND_GOVERNMENT |

For the inference part of it - when you want your custom model to determine which label corresponds to a given text -, the file format must conform with the following:

- File must contain text per line
- No header
- Format UTF-8, carriage return “\n”.

Launch data preparation with the following Terminal command. `prepare_data.py` assumes that you are at the root folder of that repository and that you have extracted the Yahoo corpus into the `yahoo_answers_csv` directory.

```shell
$> ./prepare_data.py
```

This script is tied to the Yahoo corpus and leverage the [pandas](https://pandas.pydata.org/) library to format the training and testing datasets to match Amazon Comprehend expectations described above.

Note 💡: for the moment, we encode comma characters in sentences with the equivalent HTML encoding: '&#44;'. May a better escaping exist, I did not find it in the documentation. Between double quotes doesn’t work, ‘\,’ doesn’t work neither. I opened an [issue](https://github.com/awsdocs/amazon-comprehend-developer-guide/issues/18) on the Comprehend documentation to get the recommended approach.

### Uploading the data

```shell
$> aws s3 cp comprehend-test.csv s3://hervenivon-poc/ComprehendCustomClassification/
$> aws s3 cp comprehend-train.csv s3://hervenivon-poc/ComprehendCustomClassification/
```

### Training the model

Launch the classifier training:

```shell
aws comprehend create-document-classifier --document-classifier-name "yahoo-answers" --data-access-role-arn arn:aws:iam::312306070809:role/ComprehendBucketAccessRole --input-data-config S3Uri=s3://hervenivon-poc/ComprehendCustomClassification/comprehend-train.csv --output-data-config S3Uri=s3://hervenivon-poc/ComprehendCustomClassification/TrainingOutput/ --language-code en
```

You should see something like:

```shell
{
    "DocumentClassifierArn": "arn:aws:comprehend:us-east-1:312306070809:document-classifier/yahoo-answers"
}
```

You can then track the progress with:

```shell
aws comprehend describe-document-classifier --document-classifier-arn arn:aws:comprehend:us-east-1:312306070809:document-classifier/yahoo-answers
```

You should see something like:

```shell
{
    "DocumentClassifierProperties": {
        "DocumentClassifierArn": "arn:aws:comprehend:us-east-1:312306070809:document-classifier/yahoo-answers",
        "LanguageCode": "en",
        "Status": "TRAINING",
        "SubmitTime": 1561649608.232,
        "InputDataConfig": {
            "S3Uri": "s3://hervenivon-poc/ComprehendCustomClassification/comprehend-train.csv"
        },
        "OutputDataConfig": {
            "S3Uri": "s3://hervenivon-poc/ComprehendCustomClassification/TrainingOutput/312306070809-CLR-92408cee392a4f3a83273ddd1d22bcef/output/output.tar.gz"
        },
        "DataAccessRoleArn": "arn:aws:iam::312306070809:role/ComprehendBucketAccessRole"
    }
}
```

Or when the training is finished:

```shell
{
    "DocumentClassifierProperties": {
        "DocumentClassifierArn": "arn:aws:comprehend:us-east-1:312306070809:document-classifier/yahoo-answers",
        "LanguageCode": "en",
        "Status": "TRAINED",
        "SubmitTime": 1561677325.862,
        "EndTime": 1561679052.677,
        "TrainingStartTime": 1561677482.464,
        "TrainingEndTime": 1561679043.669,
        "InputDataConfig": {
            "S3Uri": "s3://hervenivon-poc/ComprehendCustomClassification/comprehend-train.csv"
        },
        "OutputDataConfig": {
            "S3Uri": "s3://hervenivon-poc/ComprehendCustomClassification/TrainingOutput/312306070809-CLR-e53d82b1190e7d69065355d2636d80c9/output/output.tar.gz"
        },
        "ClassifierMetadata": {
            "NumberOfLabels": 10,
            "NumberOfTrainedDocuments": 989873,
            "NumberOfTestDocuments": 10000,
            "EvaluationMetrics": {
                "Accuracy": 0.7235,
                "Precision": 0.722,
                "Recall": 0.7235,
                "F1Score": 0.7219
            }
        },
        "DataAccessRoleArn": "arn:aws:iam::312306070809:role/ComprehendBucketAccessRole"
    }
}
```

In our case the training took 28 minutes.

We see that our model has a precision of 0.72—in other words, when it predicts a label, it is correct 72% of the time.

We also see that our model has a recall of 0.72—in other words, it correctly identifies 72% of labels.

### Inference

In order to launch a new job, execute the following

```shell
$> aws comprehend start-document-classification-job --document-classifier-arn arn:aws:comprehend:us-east-1:312306070809:document-classifier/yahoo-answers --input-data-config S3Uri=s3://hervenivon-poc/ComprehendCustomClassification/comprehend-test.csv,InputFormat=ONE_DOC_PER_LINE --output-data-config S3Uri=s3://hervenivon-poc/ComprehendCustomClassification/InferenceOutput/ --data-access-role-arn arn:aws:iam::312306070809:role/ComprehendBucketAccessRole
```

You should see something like this:

```shell
{
    "DocumentClassificationJobProperties": {
        "JobId": "42129ccb06ee9e7ffd74c343497c8aab",
        "JobStatus": "IN_PROGRESS",
        "SubmitTime": 1561679679.036,
        "DocumentClassifierArn": "arn:aws:comprehend:us-east-1:312306070809:document-classifier/yahoo-answers",
        "InputDataConfig": {
            "S3Uri": "s3://hervenivon-poc/ComprehendCustomClassification/comprehend-test.csv",
            "InputFormat": "ONE_DOC_PER_LINE"
        },
        "OutputDataConfig": {
            "S3Uri": "s3://hervenivon-poc/ComprehendCustomClassification/InferenceOutput/312306070809-CLN-42129ccb06ee9e7ffd74c343497c8aab/output/output.tar.gz"
        },
        "DataAccessRoleArn": "arn:aws:iam::312306070809:role/ComprehendBucketAccessRole"
    }
}
```

If you want to check the newly launched job:

```shell
$> aws comprehend describe-document-classification-job --job-id 42129ccb06ee9e7ffd74c343497c8aab
```

You should see something like:

```shell
{
    "DocumentClassificationJobProperties": {
        "JobId": "42129ccb06ee9e7ffd74c343497c8aab",
        "JobStatus": "IN_PROGRESS",
        "SubmitTime": 1561679679.036,
        "DocumentClassifierArn": "arn:aws:comprehend:us-east-1:312306070809:document-classifier/yahoo-answers",
        "InputDataConfig": {
            "S3Uri": "s3://hervenivon-poc/ComprehendCustomClassification/comprehend-test.csv",
            "InputFormat": "ONE_DOC_PER_LINE"
        },
        "OutputDataConfig": {
            "S3Uri": "s3://hervenivon-poc/ComprehendCustomClassification/InferenceOutput/312306070809-CLN-42129ccb06ee9e7ffd74c343497c8aab/output/output.tar.gz"
        },
        "DataAccessRoleArn": "arn:aws:iam::312306070809:role/ComprehendBucketAccessRole"
    }
}
```

When it is completed, `JobStatus` move to `COMPLETED`.

Then you can download the results using `OutputDataConfig.S3Uri` path:

```shell
aws s3 cp s3://hervenivon-poc/ComprehendCustomClassification/InferenceOutput/312306070809-CLN-42129ccb06ee9e7ffd74c343497c8aab/output/output.tar.gz
```

Then you can pick and choose lines in the `predictions.jsonl` file that you’ll find in the `output.tar.gz` tarball to check if you agree with your newly configured custom Amazon comprehend model.

One line from the predictions example:

```json
{"File": "comprehend-test.csv", "Line": "9", "Classes": [{"Name": "ENTERTAINMENT_AND_MUSIC", "Score": 0.9685}, {"Name": "EDUCATION_AND_REFERENCE", "Score": 0.0159}, {"Name": "BUSINESS_AND_FINANCE", "Score": 0.0102}]}
```

Which means that our custom model predicted with a 96.8% confidence score that the following text was related to the "Entertainment and music" category.

```txt
"What was the first Disney animated character to appear in color? \n Donald Duck was the first major Disney character to appear in color&#44; in his debut cartoon&#44; \"The Wise Little Hen\" in 1934.\n\nFYI: Mickey Mouse made his color debut in the 1935 'toon&#44; \"The Band Concert&#44;\" and the first color 'toon from Disney was \"Flowers and Trees&#44;\" in 1932."
```

Not that bad!
