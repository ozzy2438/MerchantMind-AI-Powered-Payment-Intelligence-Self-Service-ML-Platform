# terraform/main.tf
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "merchantmind-terraform-state"
    key            = "platform/terraform.tfstate"
    region         = "ap-southeast-2"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = "ap-southeast-2"

  default_tags {
    tags = {
      Project     = "MerchantMind"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CostCenter  = "data-ai-platform"
    }
  }
}

module "networking" {
  source = "./modules/networking"
}

module "data_lake" {
  source      = "./modules/s3-data-lake"
  environment = var.environment

  buckets = {
    raw = {
      name           = "merchantmind-raw-${var.environment}"
      versioning     = true
      lifecycle_days = 90
    }
    cleaned = {
      name           = "merchantmind-cleaned-${var.environment}"
      versioning     = true
      lifecycle_days = 180
    }
    curated = {
      name           = "merchantmind-curated-${var.environment}"
      versioning     = true
      lifecycle_days = 365
    }
    models = {
      name           = "merchantmind-models-${var.environment}"
      versioning     = true
      lifecycle_days = null
    }
    audit = {
      name           = "merchantmind-audit-${var.environment}"
      versioning     = true
      object_lock    = true
      lifecycle_days = null
    }
  }

  encryption = "aws:kms"
}

module "kinesis" {
  source          = "./modules/kinesis"
  stream_name     = "merchantmind-transactions-${var.environment}"
  shard_count     = var.environment == "production" ? 4 : 1
  retention_hours = 168
  encryption      = true
}

module "glue" {
  source = "./modules/glue"

  jobs = {
    raw_to_curated = {
      script_path     = "s3://merchantmind-models-${var.environment}/glue/raw_to_curated.py"
      schedule        = "cron(0 2 * * ? *)"
      worker_type     = "G.1X"
      num_workers     = var.environment == "production" ? 10 : 2
      timeout_minutes = 60
    }
  }

  raw_bucket     = module.data_lake.bucket_arns["raw"]
  curated_bucket = module.data_lake.bucket_arns["curated"]
}

module "feature_store" {
  source = "./modules/sagemaker-feature-store"

  feature_groups = {
    merchant_transactions = {
      record_identifier = "merchant_id"
      event_time        = "event_time"
      online_store      = true
      offline_store     = true
      offline_bucket    = module.data_lake.bucket_arns["curated"]
    }
  }
}

module "anomaly_detector" {
  source          = "./modules/sagemaker-serverless"
  model_name      = "anomaly-detector-${var.environment}"
  model_data_url  = "s3://merchantmind-models-${var.environment}/anomaly-detector/model.tar.gz"
  memory_size_mb  = 2048
  max_concurrency = var.environment == "production" ? 50 : 5
}

module "api_service" {
  source          = "./modules/ecs"
  service_name    = "merchantmind-api-${var.environment}"
  container_image = "${var.ecr_repository}:${var.image_tag}"
  desired_count   = var.environment == "production" ? 3 : 1
  cpu             = 512
  memory          = 1024

  health_check_path = "/health"

  environment_variables = {
    SAGEMAKER_ENDPOINT = module.anomaly_detector.endpoint_name
    SNOWFLAKE_ACCOUNT  = var.snowflake_account
    LOG_LEVEL          = var.environment == "production" ? "INFO" : "DEBUG"
  }

  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
}

module "monitoring" {
  source = "./modules/cloudwatch"

  dashboards = {
    platform_health = {
      widgets = [
        "api_latency_p99",
        "api_error_rate",
        "kinesis_incoming_records",
        "glue_job_status",
        "sagemaker_invocations",
        "sagemaker_model_latency",
      ]
    }
    data_quality = {
      widgets = [
        "data_freshness",
        "quality_check_pass_rate",
        "null_rate_by_column",
        "row_count_trend",
      ]
    }
    model_performance = {
      widgets = [
        "anomaly_score_distribution",
        "prediction_count",
        "drift_score",
        "feature_importance_trend",
      ]
    }
  }

  alarms = {
    api_high_latency = {
      metric    = "api_latency_p99"
      threshold = 500
      period    = 60
      action    = "sns:pagerduty"
    }
    pipeline_failure = {
      metric    = "glue_job_failures"
      threshold = 1
      period    = 300
      action    = "sns:engineering-alerts"
    }
    model_drift = {
      metric    = "data_drift_score"
      threshold = 0.3
      period    = 3600
      action    = "sns:ml-team"
    }
    data_freshness = {
      metric    = "data_staleness_seconds"
      threshold = 7200
      period    = 900
      action    = "sns:data-team"
    }
  }
}
