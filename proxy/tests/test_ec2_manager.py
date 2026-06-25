"""Tests for EC2 lifecycle management."""

import pytest
from unittest.mock import MagicMock, patch


class TestEC2Manager:
    @patch("app.ec2_manager.boto3")
    def test_get_instance_state_stopped(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "stopped"}, "PrivateIpAddress": "172.31.91.230"}]}]
        }
        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        state, ip = mgr.get_instance_state()
        assert state == "stopped"
        assert ip == "172.31.91.230"

    @patch("app.ec2_manager.boto3")
    def test_get_instance_state_running(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "running"}, "PrivateIpAddress": "172.31.1.100"}]}]
        }
        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        state, ip = mgr.get_instance_state()
        assert state == "running"
        assert ip == "172.31.1.100"

    @patch("app.ec2_manager.boto3")
    def test_start_instance(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        mgr.start_instance()
        mock_client.start_instances.assert_called_once()

    @patch("app.ec2_manager.boto3")
    def test_stop_instance(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        mgr.stop_instance()
        mock_client.stop_instances.assert_called_once()

    @patch("app.ec2_manager.boto3")
    def test_get_instance_state_no_ip(self, mock_boto3):
        """Instance in pending state may not have a private IP yet."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "pending"}}]}]
        }
        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        state, ip = mgr.get_instance_state()
        assert state == "pending"
        assert ip is None

    @patch("app.ec2_manager.boto3")
    def test_asg_get_instance_state_discovers_current_instance(self, mock_boto3):
        mock_ec2 = MagicMock()
        mock_asg = MagicMock()
        mock_boto3.client.side_effect = [mock_ec2, mock_asg]
        mock_asg.describe_auto_scaling_groups.return_value = {
            "AutoScalingGroups": [
                {
                    "DesiredCapacity": 1,
                    "Instances": [
                        {
                            "InstanceId": "i-asg-instance",
                            "LifecycleState": "InService",
                            "HealthStatus": "Healthy",
                        }
                    ],
                }
            ]
        }
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "running"}, "PrivateIpAddress": "172.31.9.10"}]}]
        }

        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        mgr._asg_name = "pdfx-backend"
        state, ip = mgr.get_instance_state()

        assert state == "running"
        assert ip == "172.31.9.10"
        assert mgr._current_instance_id == "i-asg-instance"

    @patch("app.ec2_manager.boto3")
    def test_asg_start_and_stop_scale_desired_capacity(self, mock_boto3):
        mock_ec2 = MagicMock()
        mock_asg = MagicMock()
        mock_boto3.client.side_effect = [mock_ec2, mock_asg]

        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        mgr._asg_name = "pdfx-backend"

        mgr.start_instance()
        mgr.stop_instance()

        mock_asg.set_desired_capacity.assert_any_call(
            AutoScalingGroupName="pdfx-backend",
            DesiredCapacity=1,
        )
        mock_asg.set_desired_capacity.assert_any_call(
            AutoScalingGroupName="pdfx-backend",
            DesiredCapacity=0,
        )
        mock_ec2.start_instances.assert_not_called()
        mock_ec2.stop_instances.assert_not_called()

    @patch("app.ec2_manager.boto3")
    def test_asg_mark_unhealthy_requests_replacement(self, mock_boto3):
        mock_ec2 = MagicMock()
        mock_asg = MagicMock()
        mock_boto3.client.side_effect = [mock_ec2, mock_asg]

        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        mgr._asg_name = "pdfx-backend"
        mgr._current_instance_id = "i-asg-instance"

        assert mgr.mark_unhealthy() is True
        mock_asg.set_instance_health.assert_called_once_with(
            InstanceId="i-asg-instance",
            HealthStatus="Unhealthy",
            ShouldRespectGracePeriod=False,
        )
