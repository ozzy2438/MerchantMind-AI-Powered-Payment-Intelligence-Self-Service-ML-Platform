variable "stream_name" { type = string }
variable "shard_count" { type = number }
variable "retention_hours" { type = number }
variable "encryption" { type = bool }

output "stream_name" {
  value = var.stream_name
}
