import argparse

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.image_uris import retrieve


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--model-package-group-name", type=str, required=True)
    parser.add_argument("--model-artifact-s3-uri", type=str, required=True)
    parser.add_argument("--approval-status", type=str, default="Approved")

    return parser.parse_args()


def ensure_model_package_group(sm_client, group_name, description):
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group_name)
        print(f"Model package group already exists: {group_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("ValidationException", "ResourceNotFound"):
            raise
        print(f"Creating model package group: {group_name}")
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=description,
        )
        print("Created model package group.")


def main():
    args = parse_args()

    session = boto3.Session(region_name=args.region)
    sm_client = session.client("sagemaker")

    group_description = "KKBox churn champion LightGBM model package group for controlled SageMaker validation runs"

    ensure_model_package_group(
        sm_client=sm_client,
        group_name=args.model_package_group_name,
        description=group_description,
    )

    inference_image_uri = retrieve(
        framework="sklearn",
        region=args.region,
        version="1.2-1",
        image_scope="inference",
        py_version="py3",
        instance_type="ml.m5.large",
    )

    print(f"Using inference image URI: {inference_image_uri}")

    response = sm_client.create_model_package(
        ModelPackageGroupName=args.model_package_group_name,
        ModelPackageDescription="KKBox churn champion LightGBM model registered from SageMaker training job",
        ModelApprovalStatus=args.approval_status,
        InferenceSpecification={
            "Containers": [
                {
                    "Image": inference_image_uri,
                    "ModelDataUrl": args.model_artifact_s3_uri,
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
    )

    print("\nModel package created successfully.")
    print(f"Model package ARN: {response['ModelPackageArn']}")
    print(f"Model package group: {args.model_package_group_name}")
    print("A new version has been registered in SageMaker Model Registry.")


if __name__ == "__main__":
    main()
