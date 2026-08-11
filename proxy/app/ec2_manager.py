"""Manage PDFX backend lifecycle.

The proxy supports two backend modes:

* legacy single EC2 instance mode via EC2_INSTANCE_ID
* preferred Auto Scaling mode via BACKEND_ASG_NAME

Auto Scaling mode keeps the public proxy independent of a specific instance ID.
The ASG is capped by infrastructure (normally MaxSize=1) and replaces unhealthy
instances from the launch template.
"""

import logging

import boto3

from app.config import settings

logger = logging.getLogger(__name__)


class EC2Manager:
    def __init__(self):
        self._client = boto3.client("ec2", region_name=settings.EC2_REGION)
        self._autoscaling = boto3.client("autoscaling", region_name=settings.EC2_REGION)
        self._instance_id = settings.EC2_INSTANCE_ID
        self._asg_name = settings.BACKEND_ASG_NAME
        self._current_instance_id: str | None = None

    @property
    def uses_auto_scaling(self) -> bool:
        return bool(self._asg_name)

    def get_instance_state(self) -> tuple[str, str | None]:
        """Return (state_name, private_ip) for the managed backend."""
        state, private_ip, _instance_id = self.get_instance_snapshot()
        return state, private_ip

    def get_instance_snapshot(self) -> tuple[str, str | None, str | None]:
        """Return one coherent (state, private_ip, instance_id) observation."""
        if self.uses_auto_scaling:
            return self._get_asg_instance_snapshot()

        resp = self._client.describe_instances(InstanceIds=[self._instance_id])
        inst = resp["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        ip = inst.get("PrivateIpAddress")
        self._current_instance_id = self._instance_id
        return state, ip, self._instance_id

    def start_instance(self) -> None:
        if self.uses_auto_scaling:
            logger.info("Scaling backend Auto Scaling group %s to desired capacity 1", self._asg_name)
            self._autoscaling.set_desired_capacity(
                AutoScalingGroupName=self._asg_name,
                DesiredCapacity=1,
            )
            return

        logger.info("Starting EC2 instance %s", self._instance_id)
        self._client.start_instances(InstanceIds=[self._instance_id])

    def stop_instance(self, expected_instance_id: str) -> bool:
        """Stop only the explicitly expected current backend instance.
        """
        if self.uses_auto_scaling:
            _state, _ip, current_instance_id = self.get_instance_snapshot()
            if current_instance_id != expected_instance_id:
                logger.warning(
                    "Refusing to scale backend ASG to zero: expected instance %s, current instance %s",
                    expected_instance_id,
                    current_instance_id,
                )
                return False
            logger.info("Scaling backend Auto Scaling group %s to desired capacity 0", self._asg_name)
            self._autoscaling.set_desired_capacity(
                AutoScalingGroupName=self._asg_name,
                DesiredCapacity=0,
            )
            self._current_instance_id = None
            return True

        if expected_instance_id != self._instance_id:
            logger.warning(
                "Refusing to stop legacy backend: expected instance %s, configured instance %s",
                expected_instance_id,
                self._instance_id,
            )
            return False
        logger.info("Stopping EC2 instance %s", self._instance_id)
        self._client.stop_instances(InstanceIds=[self._instance_id])
        return True

    def mark_unhealthy(self, expected_instance_id: str) -> bool:
        """Ask Auto Scaling to replace the current backend instance.

        Returns True when a replacement signal was sent. Legacy single-instance
        mode intentionally does not terminate or replace anything here.
        """
        if not self.uses_auto_scaling:
            logger.warning("Backend startup failed in legacy EC2 mode; no Auto Scaling replacement is available")
            return False

        try:
            _state, _ip, instance_id = self.get_instance_snapshot()
        except Exception as exc:
            logger.warning("Could not resolve ASG instance to mark unhealthy: %s", exc)
            return False

        if not instance_id:
            logger.warning("No ASG backend instance is available to mark unhealthy")
            return False
        if instance_id != expected_instance_id:
            logger.warning(
                "Refusing stale backend replacement request: expected instance %s, current instance %s",
                expected_instance_id,
                instance_id,
            )
            return False

        logger.error(
            "Marking ASG backend instance %s unhealthy so Auto Scaling replaces it",
            instance_id,
        )
        self._autoscaling.set_instance_health(
            InstanceId=instance_id,
            HealthStatus="Unhealthy",
            ShouldRespectGracePeriod=False,
        )
        return True

    def _get_asg_instance_snapshot(self) -> tuple[str, str | None, str | None]:
        resp = self._autoscaling.describe_auto_scaling_groups(AutoScalingGroupNames=[self._asg_name])
        groups = resp.get("AutoScalingGroups", [])
        if not groups:
            raise RuntimeError(f"Backend Auto Scaling group not found: {self._asg_name}")

        group = groups[0]
        instances = [
            inst
            for inst in group.get("Instances", [])
            if inst.get("LifecycleState") not in {"Terminating", "Terminating:Wait", "Terminating:Proceed"}
        ]
        if not instances:
            desired = int(group.get("DesiredCapacity", 0))
            self._current_instance_id = None
            return ("stopped" if desired == 0 else "pending", None, None)

        def _priority(asg_instance):
            lifecycle = asg_instance.get("LifecycleState", "")
            health = asg_instance.get("HealthStatus", "")
            if lifecycle == "InService" and health == "Healthy":
                return 0
            if lifecycle.startswith("Pending"):
                return 1
            return 2

        instance_id = sorted(instances, key=_priority)[0]["InstanceId"]
        self._current_instance_id = instance_id
        ec2_resp = self._client.describe_instances(InstanceIds=[instance_id])
        inst = ec2_resp["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        ip = inst.get("PrivateIpAddress")
        return state, ip, instance_id
