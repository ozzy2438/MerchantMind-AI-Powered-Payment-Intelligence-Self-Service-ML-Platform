output "anomaly_endpoint" {
  value = module.anomaly_detector.endpoint_name
}

output "raw_bucket_arn" {
  value = module.data_lake.bucket_arns["raw"]
}
