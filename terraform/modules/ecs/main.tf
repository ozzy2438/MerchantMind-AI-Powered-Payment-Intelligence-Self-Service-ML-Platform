variable "service_name" { type = string }
variable "container_image" { type = string }
variable "desired_count" { type = number }
variable "cpu" { type = number }
variable "memory" { type = number }
variable "health_check_path" { type = string }
variable "environment_variables" { type = map(string) }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }

output "service_name" {
  value = var.service_name
}
