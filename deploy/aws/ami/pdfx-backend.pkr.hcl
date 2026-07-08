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

variable "security_group_ids" {
  type = list(string)
  # [ephemeral SSH SG (port 22 for Packer), a SG admitted by the ECR interface
  # VPC endpoints so the build box can reach api/dkr.ecr.us-east-1].
  default = ["sg-0e29540f9db1ac31b", "sg-21ac675b"]
}

variable "backend_git_ref" {
  type = string
  # Git ref checked out into the baked AMI (the boot re-runs deploy.sh from it).
  # Empty -> provision.sh falls back to backend_image_tag. Set to a branch/SHA
  # (e.g. "main") to bake newer deploy code against an existing image tag.
  default = ""
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
  security_group_ids          = var.security_group_ids
  associate_public_ip_address = true
  ami_name                    = "pdfx-backend-baked-${var.backend_image_tag}-${local.ts}"

  # The ~200 GB AMI snapshot (7 GB image + model caches) takes far longer than
  # Packer's default AMI-ready wait; allow up to ~60 min so a slow snapshot
  # doesn't false-fail an otherwise-successful bake.
  aws_polling {
    delay_seconds = 20
    max_attempts  = 180
  }

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

  # provision.sh re-clones the repo at BACKEND_GIT_REF from GitHub, so the bake
  # only needs provision.sh itself delivered. Uploading a single file (not the
  # whole repo) avoids a slow .git upload and the root-owned-copy chmod failure.
  provisioner "file" {
    source      = "${path.root}/provision.sh"
    destination = "/tmp/provision.sh"
  }

  # Run as root, passing the bake vars explicitly (robust regardless of the
  # instance's sudoers env policy) and via `bash` so no execute bit is needed.
  provisioner "shell" {
    inline = [
      "sudo env BACKEND_IMAGE_REPO='${var.backend_image_repo}' BACKEND_IMAGE_TAG='${var.backend_image_tag}' BASE_AMI_ID='${var.base_ami_id}' AWS_REGION='${var.region}' BACKEND_GIT_REF='${var.backend_git_ref}' bash /tmp/provision.sh",
    ]
  }

  # Emit the built AMI id to manifest.json for the CI job to read (robust vs. -machine-readable).
  post-processor "manifest" {
    output     = "manifest.json"
    strip_path = true
  }
}
