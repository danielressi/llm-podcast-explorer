import os

import boto3


def write_to_r2(src_path: str, bucket, target_path: str):
    session = boto3.session.Session()
    client = session.client(
        service_name="s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
    )
    client.upload_file(src_path, bucket, target_path)


def write_to_json(json_str: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)
