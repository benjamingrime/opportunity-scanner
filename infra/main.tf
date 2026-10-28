# The running scanner: one container image on Lambda behind a Function URL, with DynamoDB for
# paused runs and the daily run counter, and S3 for large checkpoints. The LLM is chosen by var.llm.
#
#   terraform init -backend-config="bucket=opportunity-scanner-tfstate-<account id>"
#   terraform apply -var image_tag=<git sha>

terraform {
  required_version = ">= 1.10"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.70" }
  }
  backend "s3" {
    key          = "main.tfstate"
    region       = "eu-north-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = var.name } }
}

data "aws_caller_identity" "me" {}
data "aws_ecr_repository" "app" { name = var.name }

locals {
  account = data.aws_caller_identity.me.account_id
}

# --- storage ---------------------------------------------------------------------------------

resource "aws_dynamodb_table" "checkpoints" {
  name         = "${var.name}-checkpoints"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "PK"
  range_key    = "SK"
  attribute {
    name = "PK"
    type = "S"
  }
  attribute {
    name = "SK"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "usage" {
  name         = "${var.name}-usage"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "PK"
  attribute {
    name = "PK"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

resource "aws_s3_bucket" "data" {
  bucket = "${var.name}-data-${local.account}"
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "expire-checkpoints"
    status = "Enabled"
    filter { prefix = "checkpoints/" }
    expiration { days = 30 }
  }
}

# The OpenRouter key. Terraform creates the secret but never sees its value: that is set once by
# hand (see the README), so the key stays out of state files and logs.
resource "aws_secretsmanager_secret" "openrouter" {
  name                    = "${var.name}/openrouter-api-key"
  recovery_window_in_days = 7
}

# --- the function ----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "app" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = 14
}

resource "aws_iam_role" "app" {
  name = "${var.name}-lambda"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = { StringEquals = { "aws:SourceAccount" = local.account } }
    }]
  })
}

resource "aws_iam_role_policy" "app" {
  name = "app"
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat([
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.app.arn}:*"
      },
      {
        Sid      = "Traces"
        Effect   = "Allow"
        Action   = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
        Resource = "*"
      },
      {
        Sid    = "Tables"
        Effect = "Allow"
        Action = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem",
        "dynamodb:Query", "dynamodb:BatchGetItem", "dynamodb:BatchWriteItem", "dynamodb:TransactWriteItems"]
        Resource = [aws_dynamodb_table.checkpoints.arn, aws_dynamodb_table.usage.arn]
      },
      {
        Sid      = "Checkpoints"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.data.arn}/checkpoints/*"
      },
      {
        Sid      = "OpenRouterKey"
        Effect   = "Allow"
        Action   = "secretsmanager:GetSecretValue"
        Resource = aws_secretsmanager_secret.openrouter.arn
      },
      ], length(var.bedrock_models) == 0 ? [] : [{
        Sid    = "Models"
        Effect = "Allow"
        Action = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        # an inference profile (eu.*) routes to the model in several EU Regions
        Resource = flatten([for m in var.bedrock_models : [
          "arn:aws:bedrock:*::foundation-model/${trimprefix(trimprefix(m, "eu."), "global.")}",
          "arn:aws:bedrock:${var.region}:${local.account}:inference-profile/${m}",
        ]])
    }])
  })
}

resource "aws_lambda_function" "app" {
  function_name = var.name
  role          = aws_iam_role.app.arn
  package_type  = "Image"
  image_uri     = "${data.aws_ecr_repository.app.repository_url}:${var.image_tag}"
  architectures = ["arm64"]
  memory_size   = 1024
  timeout       = 300 # a 40-role CSV maps in parallel; well under this
  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.app.name
  }
  tracing_config { mode = "Active" }
  environment {
    variables = {
      SCANNER_LLM          = var.llm
      OPENROUTER_SECRET    = aws_secretsmanager_secret.openrouter.name
      SCANNER_CHECKPOINTS  = "dynamodb:${aws_dynamodb_table.checkpoints.name}"
      SCANNER_BUCKET       = aws_s3_bucket.data.bucket
      SCANNER_USAGE_TABLE  = aws_dynamodb_table.usage.name
      SCANNER_RUNS_PER_DAY = tostring(var.runs_per_day)
      SCANNER_MAX_ROLES    = tostring(var.max_roles)
      AWS_LWA_INVOKE_MODE  = "response_stream" # the page shows the agent's steps as they happen
    }
  }
}

# Public demo: the page, the API and the MCP server. Cost is held by the daily run cap, the
# project's Lambda concurrency limit and the AWS Budgets alarm.
resource "aws_lambda_function_url" "app" {
  function_name      = aws_lambda_function.app.function_name
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"
}

# a public Function URL needs both permissions: InvokeFunctionUrl alone returns 403
resource "aws_lambda_permission" "url" {
  statement_id           = "FunctionURLAllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.app.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "invoke" {
  statement_id             = "FunctionURLAllowInvoke"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.app.function_name
  principal                = "*"
  invoked_via_function_url = true
}

output "url" { value = aws_lambda_function_url.app.function_url }
