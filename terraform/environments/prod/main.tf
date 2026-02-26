module "platform" {
  source = "../.."

  environment       = "production"
  image_tag         = var.image_tag
  ecr_repository    = var.ecr_repository
  snowflake_account = var.snowflake_account
}

variable "image_tag" { type = string default = "latest" }
variable "ecr_repository" { type = string default = "merchantmind-api" }
variable "snowflake_account" { type = string default = "placeholder" }
