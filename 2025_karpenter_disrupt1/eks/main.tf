provider "aws" {
  region = "ap-northeast-2"
}

# VPC 구성
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "3.18.1"

  name = "eks-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["ap-northeast-2a"]
  public_subnets  = ["10.0.1.0/24"]

  enable_nat_gateway = false
  tags = {
    Name = "eks-vpc"
  }
}

# EKS 클러스터용 IAM 역할
module "eks_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role"
  version = "~> 5.0"

  name = "eks-cluster-role"
  assume_role_policy_json = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy",
    "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController",
  ]
}

# EKS 클러스터
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.15.0"

  cluster_name    = "my-eks-cluster"
  cluster_version = "1.31" # 최신 버전 설정

  subnets         = module.vpc.public_subnets
  vpc_id          = module.vpc.vpc_id

  cluster_endpoint_public_access = true

  cluster_iam_role_name = module.eks_role.iam_role_name

  node_groups_defaults = {
    ami_type       = "AL2_x86_64"
    capacity_type  = "SPOT" # Spot 인스턴스만 사용
    disk_size      = 20
    instance_types = ["t3.micro"] # 최소 비용의 Spot 인스턴스
    desired_capacity = 1
    min_capacity     = 1
    max_capacity     = 2
  }

  node_groups = {
    spot_nodes = {}
  }

  tags = {
    Environment = "dev"
  }
}

output "cluster_name" {
  value = module.eks.cluster_id
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  value = module.eks.cluster_security_group_id
}

