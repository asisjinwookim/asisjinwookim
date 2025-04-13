# Create the S3 bucket
resource "aws_s3_bucket" "glenn_mural_bucket_1" {
  bucket = "glenn-mural-bucket-1"
}

# # Create IAM role
resource "aws_iam_role" "s3_access_role" {
  name = "s3-access-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "s3.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# # Attach an IAM policy to the role to allow access to the S3 bucket
resource "aws_iam_policy" "s3_access_policy" {
  name   = "s3-access-policy"
  description = "Allow access to S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:ListBucket",
          "s3:GetObject",
          "s3:PutObject"
        ],
        Resource = [
          aws_s3_bucket.glenn_mural_bucket_1.arn,
          "${aws_s3_bucket.glenn_mural_bucket_1.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_s3_access_role" {
  role       = aws_iam_role.s3_access_role.name
  policy_arn = aws_iam_policy.s3_access_policy.arn
}
