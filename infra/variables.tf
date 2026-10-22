variable "name" {
  type    = string
  default = "opportunity-scanner"
}

variable "region" {
  type    = string
  default = "eu-north-1" # the project's Region; resources can't go elsewhere
}

variable "image_tag" {
  type        = string
  description = "Image tag in ECR: the git commit it was built from"
}

variable "llm" {
  type        = string
  description = "SCANNER_LLM: the model the agent uses (see scanner/llm.py). Chosen by the mapping eval"
  default     = "openrouter:anthropic/claude-haiku-5.5"
}

variable "bedrock_models" {
  type        = list(string)
  description = "Bedrock models the function may call, if var.llm is a Bedrock one"
  default     = []
}

variable "runs_per_day" {
  type    = number
  default = 200
}
