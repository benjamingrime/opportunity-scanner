# Continuous deployment without stored keys. The project's managed policies block IAM identity
# providers, so GitHub Actions can't assume a role by OIDC; instead AWS pulls from GitHub through
# a CodeConnections GitHub app, and CodeBuild deploys with its own role.
#
# Merge to main -> CodePipeline -> CodeBuild (buildspec.yml): tests, image to ECR, terraform apply
# of infra/, smoke test. The connection starts PENDING: approve it once in the console
# (Developer Tools > Settings > Connections).

variable "repository" {
  type    = string
  default = "benjamingrime/opportunity-scanner"
}

locals {
  account = data.aws_caller_identity.me.account_id
  region  = "eu-north-1"
  name    = "opportunity-scanner"
}

resource "aws_codestarconnections_connection" "github" {
  name          = "github"
  provider_type = "GitHub"
}

resource "aws_s3_bucket" "pipeline" {
  bucket        = "${local.name}-pipeline-${local.account}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "pipeline" {
  bucket                  = aws_s3_bucket.pipeline.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "pipeline" {
  bucket = aws_s3_bucket.pipeline.id
  rule {
    id     = "expire-artifacts"
    status = "Enabled"
    filter {}
    expiration { days = 14 }
  }
}

resource "aws_cloudwatch_log_group" "build" {
  name              = "/aws/codebuild/${local.name}-deploy"
  retention_in_days = 30
}

# --- CodeBuild: builds and deploys ------------------------------------------------------------

resource "aws_iam_role" "build" {
  name = "${local.name}-build"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { StringEquals = { "aws:SourceAccount" = local.account } }
    }]
  })
}

# Enough to run terraform apply on infra/ and nothing else: every resource is named opportunity-scanner*.
resource "aws_iam_role_policy" "build" {
  name = "deploy"
  role = aws_iam_role.build.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "BuildLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.build.arn}:*"
      },
      {
        Sid      = "Artifacts"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = "${aws_s3_bucket.pipeline.arn}/*"
      },
      {
        Sid      = "State"
        Effect   = "Allow"
        Action   = ["s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = [aws_s3_bucket.state.arn, "${aws_s3_bucket.state.arn}/main.tfstate*"]
      },
      {
        Sid      = "Whoami"
        Effect   = "Allow"
        Action   = ["sts:GetCallerIdentity", "ecr:GetAuthorizationToken", "logs:DescribeLogGroups"]
        Resource = "*"
      },
      {
        Sid    = "Image"
        Effect = "Allow"
        Action = ["ecr:DescribeRepositories", "ecr:DescribeImages", "ecr:BatchCheckLayerAvailability",
          "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage",
          "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:GetRepositoryPolicy", "ecr:SetRepositoryPolicy",
        "ecr:ListTagsForResource"]
        Resource = aws_ecr_repository.app.arn
      },
      {
        Sid      = "Function"
        Effect   = "Allow"
        Action   = "lambda:*"
        Resource = "arn:aws:lambda:${local.region}:${local.account}:function:${local.name}"
      },
      {
        Sid    = "FunctionRole"
        Effect = "Allow"
        Action = ["iam:GetRole", "iam:CreateRole", "iam:DeleteRole", "iam:TagRole", "iam:UntagRole",
          "iam:PutRolePolicy", "iam:GetRolePolicy", "iam:DeleteRolePolicy", "iam:ListRolePolicies",
        "iam:ListAttachedRolePolicies", "iam:ListInstanceProfilesForRole", "iam:UpdateAssumeRolePolicy", "iam:PassRole"]
        Resource = "arn:aws:iam::${local.account}:role/${local.name}-lambda"
      },
      {
        Sid      = "Tables"
        Effect   = "Allow"
        Action   = "dynamodb:*"
        Resource = "arn:aws:dynamodb:${local.region}:${local.account}:table/${local.name}-*"
      },
      {
        Sid      = "DataBucket"
        Effect   = "Allow"
        Action   = "s3:*"
        Resource = ["arn:aws:s3:::${local.name}-data-${local.account}", "arn:aws:s3:::${local.name}-data-${local.account}/*"]
      },
      {
        Sid      = "FunctionLogs"
        Effect   = "Allow"
        Action   = "logs:*"
        Resource = "arn:aws:logs:${local.region}:${local.account}:log-group:/aws/lambda/${local.name}*"
      },
      {
        Sid    = "Secret"
        Effect = "Allow"
        Action = ["secretsmanager:CreateSecret", "secretsmanager:DescribeSecret", "secretsmanager:DeleteSecret",
        "secretsmanager:GetResourcePolicy", "secretsmanager:TagResource", "secretsmanager:UntagResource"]
        Resource = "arn:aws:secretsmanager:${local.region}:${local.account}:secret:${local.name}/*"
      },
    ]
  })
}

resource "aws_codebuild_project" "deploy" {
  name          = "${local.name}-deploy"
  service_role  = aws_iam_role.build.arn
  build_timeout = 30
  source {
    type      = "CODEPIPELINE"
    buildspec = "buildspec.yml"
  }
  artifacts { type = "CODEPIPELINE" }
  environment {
    type            = "ARM_CONTAINER"
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/amazonlinux-aarch64-standard:3.0"
    privileged_mode = true # docker build
  }
  logs_config {
    cloudwatch_logs { group_name = aws_cloudwatch_log_group.build.name }
  }
}

# --- CodePipeline: on each push to main -------------------------------------------------------

resource "aws_iam_role" "pipeline" {
  name = "${local.name}-pipeline"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codepipeline.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { StringEquals = { "aws:SourceAccount" = local.account } }
    }]
  })
}

resource "aws_iam_role_policy" "pipeline" {
  name = "pipeline"
  role = aws_iam_role.pipeline.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:GetBucketVersioning"]
        Resource = [aws_s3_bucket.pipeline.arn, "${aws_s3_bucket.pipeline.arn}/*"]
      },
      # without the conditions, the pipeline could read every repository the GitHub app can see.
      # The connection's ARN uses the older codestar-connections prefix, so both forms are granted.
      {
        Effect    = "Allow"
        Action    = "codeconnections:UseConnection"
        Resource  = aws_codestarconnections_connection.github.arn
        Condition = { StringEquals = { "codeconnections:FullRepositoryId" = var.repository } }
      },
      {
        Effect    = "Allow"
        Action    = "codestar-connections:UseConnection"
        Resource  = aws_codestarconnections_connection.github.arn
        Condition = { StringEquals = { "codestar-connections:FullRepositoryId" = var.repository } }
      },
      {
        Effect   = "Allow"
        Action   = ["codebuild:StartBuild", "codebuild:BatchGetBuilds"]
        Resource = aws_codebuild_project.deploy.arn
      },
    ]
  })
}

resource "aws_codepipeline" "deploy" {
  name           = "${local.name}-deploy"
  role_arn       = aws_iam_role.pipeline.arn
  pipeline_type  = "V2"
  execution_mode = "QUEUED" # deploys run one at a time, in order

  artifact_store {
    location = aws_s3_bucket.pipeline.bucket
    type     = "S3"
  }

  stage {
    name = "Source"
    action {
      name             = "GitHub"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeStarSourceConnection"
      version          = "1"
      namespace        = "Source"
      output_artifacts = ["source"]
      configuration = {
        ConnectionArn    = aws_codestarconnections_connection.github.arn
        FullRepositoryId = var.repository
        BranchName       = "main"
      }
    }
  }

  stage {
    name = "Deploy"
    action {
      name            = "TestBuildDeploy"
      category        = "Build"
      owner           = "AWS"
      provider        = "CodeBuild"
      version         = "1"
      input_artifacts = ["source"]
      configuration = {
        ProjectName = aws_codebuild_project.deploy.name
        EnvironmentVariables = jsonencode([
          { name = "COMMIT", value = "#{Source.CommitId}", type = "PLAINTEXT" },
        ])
      }
    }
  }

  trigger {
    provider_type = "CodeStarSourceConnection"
    git_configuration {
      source_action_name = "GitHub"
      push {
        branches { includes = ["main"] }
      }
    }
  }
}

output "connection_arn" { value = aws_codestarconnections_connection.github.arn }
output "connection_status" { value = aws_codestarconnections_connection.github.connection_status }
