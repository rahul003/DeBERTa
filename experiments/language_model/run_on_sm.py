import os
from sagemaker.pytorch import PyTorch
import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import HuggingFace
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
import boto3

role = (
    get_execution_role()
)  # provide a pre-existing role ARN as an alternative to creating a new role
print(f"SageMaker Execution Role: {role}")

client = boto3.client("sts")
account = client.get_caller_identity()["Account"]
print(f"AWS account: {account}")

session = boto3.session.Session()
region = session.region_name
print(f"AWS region: {region}")

sm_boto_client = boto3.client("sagemaker")
sagemaker_session = sagemaker.session.Session(boto_session=session)

# get default bucket
default_bucket = sagemaker_session.default_bucket()
print()
print("Default bucket for this session: ", default_bucket)

s3_output_location = f"s3://{default_bucket}/output/"

mpioptions = "-x NCCL_DEBUG=WARN -x SMDEBUG_LOG_LEVEL=ERROR "
mpioptions += "-x FI_EFA_USE_DEVICE_RDMA=1 -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1"

metric_definitions = [
    {"Name": "base_metric", "Regex": "<><><><><><>"}
]

instance_type = "ml.p3.16xlarge"
instance_count = 1
processes_per_host = 8

volume_size = 500
base_job_name = f'deberta-v3'

checkpoint_bucket = f"s3://sagemaker-{region}-{account}/"
checkpoint_s3_uri = (
        f"{checkpoint_bucket}/experiments/deberta-v3/{base_job_name}/"
    )

estimator = PyTorch(
        'launch.py', 
        source_dir=os.getcwd(),
        role=role,
        instance_type=instance_type,
        volume_size=volume_size,
        instance_count=instance_count,
        sagemaker_session=sagemaker_session,
        distribution={
            "mpi": {
                "enabled": True,
                "processes_per_host": processes_per_host,
                "custom_mpi_options": mpioptions,
            },
        },
        framework_version="1.8.1",
        py_version="py36",
        output_path=s3_output_location,
   #     checkpoint_s3_uri=checkpoint_s3_uri,
   #     checkpoint_local_path=hyperparameters["checkpoint-dir"] if use_fsx else None,
        metric_definitions=metric_definitions,
   #     hyperparameters=hyperparameters,
        debugger_hook_config=False,
        disable_profiler=True,
        base_job_name=base_job_name,
    )
estimator.fit(logs=True)
