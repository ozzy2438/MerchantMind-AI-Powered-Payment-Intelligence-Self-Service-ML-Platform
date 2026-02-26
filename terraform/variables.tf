variable "environment" {
  type    = string
  default = "staging"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "ecr_repository" {
  type    = string
  default = "merchantmind-api"
}

variable "snowflake_account" {
  type    = string
  default = "placeholder"
}
