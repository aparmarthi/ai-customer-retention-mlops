import argparse
from pathlib import Path
import json
import time

import boto3
from sagemaker.core.common_utils import name_from_base
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.image_uris import retrieve
from sagemaker.core.training.configs import (
    Compute,
    InputData,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train import ModelTrainer

_SCRIPT_DIR = str(Path(__file__).resolve().parent)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--train-s3-uri", type=str, required=True)

    parser.add_argument("--instance-type", type=str, default="ml.m5.large")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--max-run", type=int, default=3600)
    parser.add_argument("--subset-fraction", type=float, default=1.0)
    parser.add_argument("--wait", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = Session(
        boto_session=boto_session,
        default_bucket=args.bucket,
    )

    training_image = retrieve(
        framework="sklearn",
        region=args.region,
        version="1.2-1",
        image_scope="training",
        py_version="py3",
        instance_type=args.instance_type,
    )

    job_name = name_from_base("kkbox-churn-champion")
    output_path = f"s3://{args.bucket}/kkbox-churn/training/artifacts/"

    print("Launching SageMaker training job with:")
    print(f"  region:            {args.region}")
    print(f"  role arn:          {args.role_arn}")
    print(f"  bucket:            {args.bucket}")
    print(f"  train s3 uri:      {args.train_s3_uri}")
    print(f"  instance type:     {args.instance_type}")
    print(f"  instance count:    {args.instance_count}")
    print(f"  max run (seconds): {args.max_run}")
    print(f"  subset fraction:   {args.subset_fraction}")
    print(f"  output path:       {output_path}")
    print(f"  training image:    {training_image}")
    print(f"  job name:          {job_name}")

    trainer = ModelTrainer(
        training_image=training_image,
        source_code=SourceCode(
            source_dir=_SCRIPT_DIR,
            entry_script="train.py",
            requirements="requirements.txt",
        ),
        role=args.role_arn,
        base_job_name="kkbox-churn-champion",
        compute=Compute(
            instance_type=args.instance_type,
            instance_count=args.instance_count,
        ),
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=args.max_run,
        ),
        output_data_config=OutputDataConfig(
            s3_output_path=output_path,
        ),
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "target-col": "is_churn",
            "time-col": "txn_last_date",
            "id-col": "msno",
            "subset-fraction": str(args.subset_fraction),
            "random-state": "42",
        },
    )

    train_data = InputData(
        channel_name="train",
        data_source=args.train_s3_uri,
    )

    # #region agent log
    with open("debug-cbacb8.log", "a") as _f:
        _f.write(
            json.dumps(
                {
                    "sessionId": "cbacb8",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "cloud/sagemaker/launch_training_job.py:105",
                    "message": "About to call trainer.train",
                    "data": {
                        "instance_type": args.instance_type,
                        "instance_count": args.instance_count,
                        "region": args.region,
                    },
                    "timestamp": int(time.time() * 1000),
                }
            )
            + "\n"
        )
    # #endregion agent log

    try:
        trainer.train(
            input_data_config=[train_data],
            wait=args.wait,
            logs=args.wait,
        )
    except Exception as e:
        # #region agent log
        with open("debug-cbacb8.log", "a") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "cbacb8",
                        "runId": "pre-fix",
                        "hypothesisId": "H1",
                        "location": "cloud/sagemaker/launch_training_job.py:113",
                        "message": "trainer.train raised exception",
                        "data": {
                            "error_type": type(e).__name__,
                            "error_str": str(e),
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
        # #endregion agent log
        raise

    print("\nTraining job submitted successfully.")
    print(f"Training job name: {job_name}")
    print(f"Artifacts will land under: {output_path}")
    print("\nAfter the job succeeds, your model artifact should be here:")
    print(
        f"s3://{args.bucket}/kkbox-churn/training/artifacts/{job_name}/output/model.tar.gz"
    )


if __name__ == "__main__":
    main()
