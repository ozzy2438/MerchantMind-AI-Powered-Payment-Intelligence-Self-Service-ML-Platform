variable "jobs" {
  type = map(object({
    script_path     = string
    schedule        = string
    worker_type     = string
    num_workers     = number
    timeout_minutes = number
  }))
}

variable "raw_bucket" { type = string }
variable "curated_bucket" { type = string }

output "job_names" {
  value = keys(var.jobs)
}
