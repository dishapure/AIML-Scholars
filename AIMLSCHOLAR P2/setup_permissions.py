"""Grant the deployed agent access to this project's KB, memory and browser.

Run locally after `agentcore deploy`, using the student's AWS credentials.
Reads main.py as text; never imports or runs the student's agent implementation.
"""

import argparse
import ast
import json
from pathlib import Path
import re
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import yaml

POLICY_NAME = "CustomerSupportIntegrations"


def read_settings(source):
    settings = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        for name in names:
            if name in {"KB_ID", "MEMORY_ID", "REGION"}:
                try:
                    settings[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    settings[name] = None
    patterns = {
        "KB_ID": r"[A-Za-z0-9]{10}",
        "MEMORY_ID": r"[A-Za-z][A-Za-z0-9_]{0,99}-[A-Za-z0-9]{10}",
        "REGION": r"[a-z]{2}(?:-[a-z]+)+-\d+",
    }
    for name, pattern in patterns.items():
        value = settings.get(name)
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"Set {name} to your resource's literal string value in main.py first.")
    return settings


def deployment(config, agent_name, region):
    if not isinstance(config, dict):
        raise ValueError("Deployment configuration must be a YAML mapping.")
    name = agent_name or config.get("default_agent")
    agent = config.get("agents", {}).get(name)
    if not isinstance(agent, dict):
        raise ValueError("Select a configured agent with --agent, or run agentcore configure first.")
    aws = agent.get("aws", {})
    role = aws.get("execution_role") or ""
    match = re.fullmatch(r"arn:([a-z0-9-]+):iam::(\d{12}):role/(.+)", role)
    runtime = agent.get("bedrock_agentcore", {}).get("agent_id")
    if not match or not runtime:
        raise ValueError("Run agentcore deploy first; the runtime ID and execution-role ARN are required.")
    partition, account, role_path = match.groups()
    if aws.get("region") != region or str(aws.get("account")) != account:
        raise ValueError("main.py region and deployment account/region do not match the execution role.")
    return name, partition, account, role, role_path.rsplit("/", 1)[-1], runtime


def build_policy(partition, account, settings):
    region = settings["REGION"]
    memory = f"arn:{partition}:bedrock-agentcore:{region}:{account}:memory/{settings['MEMORY_ID']}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "bedrock:Retrieve",
                "Resource": f"arn:{partition}:bedrock:{region}:{account}:knowledge-base/{settings['KB_ID']}",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:GetMemory",
                    "bedrock-agentcore:RetrieveMemoryRecords",
                    "bedrock-agentcore:CreateEvent",
                ],
                "Resource": [memory, memory + "/*"],
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock-agentcore:StartBrowserSession",
                    "bedrock-agentcore:StopBrowserSession",
                    "bedrock-agentcore:GetBrowserSession",
                    "bedrock-agentcore:UpdateBrowserStream",
                    "bedrock-agentcore:ConnectBrowserAutomationStream",
                ],
                "Resource": f"arn:{partition}:bedrock-agentcore:{region}:aws:browser/aws.browser.v1",
            },
        ],
    }


def apply_policy(session, account, role_arn, role_name, runtime_id, settings, policy):
    if session.client("sts").get_caller_identity()["Account"] != account:
        raise ValueError("Your AWS credentials belong to a different account than this deployment.")
    control = session.client("bedrock-agentcore-control")
    runtime = control.get_agent_runtime(agentRuntimeId=runtime_id)
    if runtime["roleArn"] != role_arn:
        raise ValueError("The deployed runtime uses a different role. Refresh the deployment configuration.")
    # Verify both resource IDs in this account/region before changing any permissions.
    session.client("bedrock-agent").get_knowledge_base(knowledgeBaseId=settings["KB_ID"])
    control.get_memory(memoryId=settings["MEMORY_ID"])
    session.client("iam").put_role_policy(
        RoleName=role_name, PolicyName=POLICY_NAME, PolicyDocument=json.dumps(policy)
    )


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / ".bedrock_agentcore.yaml")
    parser.add_argument("--main", type=Path, default=root / "main.py", help="Agent source containing resource IDs")
    parser.add_argument("--agent", help="Agent name; defaults to default_agent in the deployment config")
    parser.add_argument("--profile", help="Optional AWS CLI profile; otherwise use the normal AWS credentials")
    parser.add_argument("--dry-run", action="store_true", help="Print the policy without calling AWS")
    args = parser.parse_args(argv)
    try:
        settings = read_settings(args.main.read_text(encoding="utf-8"))
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        name, partition, account, role, role_name, runtime = deployment(config, args.agent, settings["REGION"])
        policy = build_policy(partition, account, settings)
        print(f"Agent: {name}\nExecution role: {role}\nInline policy: {POLICY_NAME}")
        if args.dry_run:
            print(json.dumps(policy, indent=2))
            print("Preview only; no AWS calls or changes made.")
            return 0
        session = boto3.Session(profile_name=args.profile, region_name=settings["REGION"])
        apply_policy(session, account, role, role_name, runtime, settings, policy)
        print("Permissions applied. Wait briefly for IAM propagation, then invoke your agent.")
        print(f"Rerunning updates only {POLICY_NAME}; other role policies are preserved.")
        return 0
    except (OSError, SyntaxError, ValueError, yaml.YAMLError, BotoCoreError, ClientError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())