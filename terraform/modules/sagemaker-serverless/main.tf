variable "model_name" { type = string }
variable "model_data_url" { type = string }
variable "memory_size_mb" { type = number }
variable "max_concurrency" { type = number }

output "endpoint_name" {
  value = "${var.model_name}-endpoint"
}
