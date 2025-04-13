# IAM User
# Format: An User --assumes--> A Role <--attached-- A Policy
# Example: glenn_mural --assumes--> terraformer <--attached-- terraformer_policy

resource "aws_iam_user" "glenn_mural" {
  name = "glenn.mural"
  path = "/"
}

resource "aws_iam_role" "terraformer" {
  name = "terraformer"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          AWS = "arn:aws:iam::${var.caller_account_id}:user/${aws_iam_user.glenn_mural.name}"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Create policies that allow the Terraformer Role to use Terraform commands
resource "aws_iam_policy" "terraformer_policy" {
  name        = "terraformer_policy"
  description = "terraformer_policy"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      # Policy to allow terraform import, plan, apply for limited services
		  {
		  	"Effect" = "Allow",
		  	"Action" = [
		  		"sts:AssumeRole",
          "iam:Get*",
          "iam:List*",
		  	],
		  	"Resource" = ["arn:aws:iam::${var.caller_account_id}:user/glenn.mural"]
		  },
		  {
		  	"Effect" = "Allow",
		  	"Action" = [
		  		"iam:*"
		  	],
		  	"Resource" = [
          "arn:aws:iam::${var.caller_account_id}:role/terraformer",
          "arn:aws:iam::${var.caller_account_id}:policy/terraformer_policy",
          "arn:aws:iam::${var.caller_account_id}:policy/s3-access-policy",
          "arn:aws:iam::${var.caller_account_id}:role/s3-access-role",
        ]
		  },
		  {
		  	"Effect" = "Allow",
		  	"Action" = [
		  		"s3:*"
		  	],
		  	"Resource" = [
          "arn:aws:s3:::glenn-mural-bucket-1",
          "arn:aws:s3:::glenn-mural-bucket-1/*",
        ]
		  }
    ]
  })

  lifecycle {
    ignore_changes = [
      description
     ]
  }
}

# Attach the policy to the Terraformer Role
resource "aws_iam_role_policy_attachment" "attach_terraformer_policy" {
  role       = aws_iam_role.terraformer.name
  policy_arn = aws_iam_policy.terraformer_policy.arn
}
