# Amazon Comprehend Experiment

<a id="Purpose"></a>
## Purpose 🎯

This repository provides resources to quickly analyze text and build a custom text classifier able to assign a specific class to a given text. It relates to the NLP (Natural Language Processing) field.

<a id="Services"></a>
## AWS Services ☁️

This repository explores [Amazon Comprehend](https://aws.amazon.com/comprehend/), a natural language processing (NLP) service that uses machine learning (ML) to find insights and relationships in texts. Amazon Comprehend identifies the language of the text; extracts key phrases, places, people, brands, or events; and understands how positive or negative the text is. For more information about everything Amazon Comprehend can do, see [Amazon Comprehend Features](https://aws.amazon.com/comprehend/features/).

In order to support that experiments other Amazon services are leveraged:

Amazon S3 to store the dataset for training and asynchronous analysis.

>[Amazon Simple Storage Service](https://aws.amazon.com/s3/getting-started/) is storage for the Internet. It is designed to make web-scale computing easier for developers. Amazon S3 has a simple web services interface that you can use to store and retrieve any amount of data, at any time, from anywhere on the web

Amazon Sagemaker Notebook Instances to get an integrated to AWS environment able to manipulate and explore data:

> [Amazon Sagemaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html) provides an integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis, so you don't have to manage servers.

Amazon API Gateway and AWS Lambda to build a serverless API:

> [Amazon API Gateway](https://aws.amazon.com/api-gateway/getting-started/) is a fully managed service that makes it easy for developers to create, publish, maintain, monitor, and secure APIs at any scale. APIs act as the "front door" for applications to access data, business logic, or functionality from your backend services.

> [AWS Lambda](https://aws.amazon.com/lambda/getting-started/) lets you run code without provisioning or managing servers. You pay only for the compute time you consume.

<a id="Dataset"></a>
## Data and labels 🗄

We are going to use Yahoo answers corpus used in “[Text Understanding from Scratch](https://arxiv.org/abs/1502.01710)” paper by Xiang Zhang and Yann LeCun. This dataset is made available on the [AWS Open Data Registry](https://registry.opendata.aws/fast-ai-nlp/).

The guided steps aim at helping you using your own dataset to train your Custom Classifier, follow detailed recommendations, they are here to help you.

<a id="Repository"></a>
## Repository structure 🏗

```bash
.
├── README.md                                             <-- This file
├── comprehend.ipynb                                      <-- Jupyter notebook which provides all details to interact with Amazon Comprehend
├── comprehend-experiment-notebook.yml                    <-- Cloud formation template to deploy the notebook
├── sam-app                                               <-- To support a real-time analysis API open to others in the Jupyter notebook, this repository also provides a SAM application to quickly deploy this API
│   ├── README.md
│   ├── custom_classifier                                  <-- Lambda function code
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── events
│   │   └── event.json                                    <-- event to test the API
│   ├── template.yaml                                     <-- AWS SAM template
│   └── tests                                             <-- Unit test for the lambda function
├── images                                                <-- Images used in the jupyther notebook. Some are drawio based and can be edited with xcode + drawio extension
└── command-line-path                                     <-- Directory for creating a Custom Classifier from the AWS CLI
    ├── ComprehendBucketAccessRole-Permissions.json       <-- Permissions for Amazon Comprehend to read the bucket
    ├── ComprehendBucketAccessRole-TrustPolicy.json       <-- Trust Policy for Amazon Comprehend to read the bucket
    ├── README.md                                         <-- Detailed step by step for the command line version
    ├── prepare_data.py                                   <-- Python script for data preparation
    └── requirements.txt                                  <-- Python script dependencies
```

## Prerequisites ⚙️

You have an [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html), and the AWS CLI is [installed and configured](https://docs.aws.amazon.com/cli/latest/userguide/install-macos.html). You have the proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup to run to both create and run a Sagemaker notebook instance.

## Deploy the Sagemaker notebook instance

```bash
aws cloudformation deploy --template-file comprehend-experiment-notebook.yml --stack-name comprehend-experiment
```

## Note ℹ

Although the `comprehend.ipynb` notebook has been built to run in an [Amazon SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html), you should be able to run it outside of a notebook instance with minimal modifications (updating IAM role definition and installing the necessary libraries).
