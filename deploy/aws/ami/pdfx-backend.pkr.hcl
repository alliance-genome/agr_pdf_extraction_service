# deploy/aws/ami/pdfx-backend.pkr.hcl
packer {
  required_plugins {
    amazon = { source = "github.com/hashicorp/amazon", version = ">= 1.3.0" }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "base_ami_id" {
  type = string
}

variable "backend_image_repo" {
  type = string # 100225593120.dkr.ecr.us-east-1.amazonaws.com/agr_pdfx_backend
}

variable "backend_image_tag" {
  type = string # immutable merge SHA
}

variable "build_instance_type" {
  type    = string
  default = "g6.2xlarge"
}

variable "iam_instance_profile" {
  type = string # profile granting ECR read on the build box
}

variable "subnet_id" {
  type = string
}

variable "root_volume_size" {
  type    = number
  default = 200
}

locals { ts = formatdate("YYYYMMDD-hhmmss", timestamp()) }

source "amazon-ebs" "pdfx_backend" {
  region                      = var.region
  source_ami                  = var.base_ami_id
  instance_type               = var.build_instance_type
  ssh_username                = "ec2-user"
  iam_instance_profile        = var.iam_instance_profile
  subnet_id                   = var.subnet_id
  associate_public_ip_address = true
  ami_name                    = "pdfx-backend-baked-${var.backend_image_tag}-${local.ts}"

  launch_block_device_mappings {
    device_name           = "/dev/xvda"
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = {
    Project         = "pdfx"
    Role            = "backend-baked"
    BackendImageTag = var.backend_image_tag
    BaseAmiId       = var.base_ami_id
  }
}

build {
  sources = ["source.amazon-ebs.pdfx_backend"]

  # Pre-create the upload destination so the file provisioner copies repo
  # contents into it (trailing-slash source => contents, not the dir itself).
  provisioner "shell" {
    inline = ["mkdir -p /tmp/repo"]
  }

  # Ship the repo scripts the provisioner needs.
  provisioner "file" {
    source      = "${path.root}/../../../" # repo root
    destination = "/tmp/repo"
  }

  provisioner "shell" {
    environment_vars = [
      "BACKEND_IMAGE_REPO=${var.backend_image_repo}",
      "BACKEND_IMAGE_TAG=${var.backend_image_tag}",
      "BASE_AMI_ID=${var.base_ami_id}",
      "AWS_REGION=${var.region}",
      "BACKEND_GIT_REF=${var.backend_image_tag}",
    ]
    inline = [
      "sudo cp -r /tmp/repo /home/ec2-user/agr_pdf_extraction_service_src || true",
      "cd /home/ec2-user/agr_pdf_extraction_service_src || cd /tmp/repo",
      "chmod +x deploy/aws/ami/provision.sh",
      "sudo -E deploy/aws/ami/provision.sh",
    ]
  }

  # Emit the built AMI id to manifest.json for the CI job to read (robust vs. -machine-readable).
  post-processor "manifest" {
    output     = "manifest.json"
    strip_path = true
  }
}
