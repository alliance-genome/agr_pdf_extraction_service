"""Tests for EC2 lifecycle management."""

import pytest
from unittest.mock import MagicMock, patch


class TestEC2Manager:
    @patch("app.ec2_manager.boto3")
    def test_get_instance_state_stopped(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_instances.return_value = {
            "Reservations": [{"Instances": [{"State": {"Name": "stopped"}, "PrivateIpAddress": "REDACTED-IP"}]}]
        }
        from app.ec2_manager import EC2Manager
        mgr = EC2Manager()
        state, ip = mgr.get_instance_state()
        assert state == "stopped"
        assert ip == "REDACTED-IP"

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
