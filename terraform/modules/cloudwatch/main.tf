variable "dashboards" {
  type = map(object({
    widgets = list(string)
  }))
}

variable "alarms" {
  type = map(object({
    metric    = string
    threshold = number
    period    = number
    action    = string
  }))
}

output "dashboard_names" {
  value = keys(var.dashboards)
}
