provider "aws" {
  region = "us-east-1"

  # Use the role to assume (terraformer_role)
  assume_role {
    role_arn     = "arn:aws:iam::${var.caller_account_id}:role/terraformer"
  }
}

