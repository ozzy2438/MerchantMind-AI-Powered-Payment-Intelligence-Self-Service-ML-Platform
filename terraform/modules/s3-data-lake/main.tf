variable "environment" {
  type = string
}

variable "buckets" {
  type = map(object({
    name           = string
    versioning     = bool
    lifecycle_days = number
    object_lock    = optional(bool)
  }))
}

variable "encryption" {
  type = string
}

output "bucket_arns" {
  value = { for k, v in var.buckets : k => "arn:aws:s3:::${v.name}" }
}
