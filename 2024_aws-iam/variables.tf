# Example usage
# 1. `export TF_VAR_caller_account_id={your aws account id}` and `terraform plan`
# 2. `TF_VAR_caller_account_id={your aws account id} terraform plan`
variable "caller_account_id" {
  type    = string
  default = ""
}

