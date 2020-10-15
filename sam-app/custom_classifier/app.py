import boto3
import json
import os

# Init
client = boto3.client('comprehend')
endpoint_arn = os.environ['ENDPOINT_ARN']

def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    
    try:
        print('QueryStringParameter: {}'.format(event['queryStringParameters']))
        if event['queryStringParameters'] == None or not 'text' in event['queryStringParameters']:
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Please provide a text parameter as a querystring.'}),
            }
        text = event['queryStringParameters']['text']
        if len(text) <= 5:
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'Text length must be superior to 5 caracters.'}),
            }
        response = client.classify_document(
            Text=text,
            EndpointArn=endpoint_arn
        )
        return {
            'statusCode': 200,
            'body': json.dumps(response['Classes']),
        }
    except Exception as e:
        print(e)
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'An error occured, please try again later.'}),
        }