# Applied by hand, rarely: the Terraform state bucket, the image repository (which must exist before
# the main stack: its Lambda needs an image) and the deploy pipeline (pipeline.tf).
#
#   first time: comment out the backend block, terraform init && terraform apply, then restore it and
#   terraform init -migrate-state -backend-config="bucket=opportunity-scanner-tfstate-<account id>"
#   after that: terraform init -backend-config="bucket=..." && terraform apply

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.70" }
  }
  backend "s3" {
    key          = "bootstrap.tfstate"
    region       = "eu-north-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = "eu-north-1"
  default_tags { tags = { project = "opportunity-scanner" } }
}

data "aws_caller_identity" "me" {}

resource "aws_s3_bucket" "state" {
  bucket = "opportunity-scanner-tfstate-${data.aws_caller_identity.me.account_id}"
}

resource "aws_s3_bucket_versioning" "state" {
  bucket = aws_s3_bucket.state.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "state" {
  bucket                  = aws_s3_bucket.state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_ecr_repository" "app" {
  name                 = "opportunity-scanner"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({ rules = [{
    rulePriority = 1, description = "keep the last 10 images"
    selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
    action       = { type = "expire" }
  }] })
}

output "state_bucket" { value = aws_s3_bucket.state.bucket }
output "repository_url" { value = aws_ecr_repository.app.repository_url }
