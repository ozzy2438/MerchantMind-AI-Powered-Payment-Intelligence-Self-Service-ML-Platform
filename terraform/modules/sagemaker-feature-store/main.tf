variable "feature_groups" {
  type = map(object({
    record_identifier = string
    event_time        = string
    online_store      = bool
    offline_store     = bool
    offline_bucket    = string
  }))
}

output "feature_group_names" {
  value = keys(var.feature_groups)
}
