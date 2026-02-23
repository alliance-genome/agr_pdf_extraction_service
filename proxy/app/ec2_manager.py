"""Manage EC2 instance lifecycle: start, stop, describe."""

import logging

import boto3

from app.config import settings

logger = logging.getLogger(__name__)


class EC2Manager:
    def __init__(self):
        self._client = boto3.client("ec2", region_name=settings.EC2_REGION)
        self._instance_id = settings.EC2_INSTANCE_ID

    def get_instance_state(self) -> tuple[str, str | None]:
        """Return (state_name, private_ip) for the managed instance."""
        resp = self._client.describe_instances(InstanceIds=[self._instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        ip = inst.get("PrivateIpAddress")
        return state, ip

    def start_instance(self) -> None:
        logger.info("Starting EC2 instance %s", self._instance_id)
        self._client.start_instances(InstanceIds=[self._instance_id])

    def stop_instance(self) -> None:
        logger.info("Stopping EC2 instance %s", self._instance_id)
        self._client.stop_instances(InstanceIds=[self._instance_id])
