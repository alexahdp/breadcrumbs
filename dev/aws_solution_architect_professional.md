## AWS Solutions Architect professional course notes

### IAM
 - Users  
 - Groups  
 - Roles  
   - EC2 instance Roles  
   - Service Roles  
   - Cross Account roles
 - Policies  
   - AWS managed  
   - Customer managed  
   - Inline policies  
 - Resource based policies  

When you need to provide access to user from another account, you provide possibility to assume a role  

**Suitable services:**  
 - *Access Advisor* - see permissions granted and when last accessed  
 - *Access Analyzer* - alanyze resources that are shared with external entity  

**IAM Access Analyzer**
(Can be used as a security tool)  
  - analyze resources that are shared with external entity  
  - analyze policies for S3 buckets, SQS queues, KMS keys, IAM roles  
  - analyze policies for resources in your account  
  - analyze policies for resources in another account  
  - analyze policies for resources in organization  
Access Analyzer can analyze logs in CloudTrail and generate a policy with fine-grained permissions to appropriate services  


**IAM Access Advisor**
  - see permissions granted and when last accessed  
  - see services that have not been accessed  
  - see when last accessed  
  - see when last MFA was used  
  - validates your policy against IAM policy grammar  
  

**IAM Permission Boundaries**
  - work for users and roles (not for groups)  
  - set of permissions that limit the maximum permissions that can be applied to an IAM entity
  - used to limit permissions to users or roles
  - Use case examples:
    - allow developers to self-assign policies and manage their own permissions, while making sure they cannot escalate their own privileges
    - allow users to create roles in their own accounts, but limit the permissions that can be delegated to those roles


**IAM Security Tools**
  - IAM Credentials Report
  - IAM Access Advisor
  - IAM Access Analyzer
  - IAM Policy Simulator
  - IAM Policy Validator
  - IAM Policy Visual Editor
  - IAM Access Analyzer Visual Editor

*Next resources can be shared with other accounts:*  
  - S3 buckets  
  - SQS queues  
  - SNS topics  
  - KMS keys  
  - IAM roles  
  - Lamda functions and Layers  
  - Secrets Manager secrets  


**STS**
Allows to assume roles (in the same account and between accounts)
  - AssumeRole
  - AssumeRoleWithSAML
  - AssumeRoleWithWebIdentity
  - GetFederationToken
  - GetSessionToken

*Assume Role confused deputy* - occurs when there are three accounts:
 - Account A - the account which is posesses the role
 - Account B - the account which is allowed to assume the role
 - Account C - the account which is not allowed to assume the role
Account C can send a request to Account B to assume the role in Account A. Account B trusts Account C and allows it to assume the role in Account A. Therefore, Account C can now assume the role in Account A.
To mitigate this problem, Account A provides not only Role ARN, but also External ID. Account C must provide this External ID when assuming the role. Account B must validate the External ID and allow or deny the request based on the External ID.  

Also we can use tags to make access privileges more granural. For example, when we have a role that allows access to all S3 buckets, we can add a tag to the role and allow access only to buckets with this tag (via bucket policy checking the tag).

**Identity Federation** - allows to use external identity provider (IdP) to grant access to AWS resources.  
  - SAML 2.0  
  - OpenID Connect  
  - Custom Identity Broker  
  - SSO  

Use-Cases:  
 - a corporate has its own IdP and wants to allow employees to access AWS resources using their corporate credentials  
 - web/mobile application that needs access to AWS resources  


### Active Directory
 - **AWS Managed Microsoft AD** - full AD on AWS  
   - requires two subnets in two AZs (at least two AZs)  
   - requires two domain controllers in two AZs  
   - requires EC2 instances to be joined to the domain (with Windows)  
   - can be standalone or joined to on-premise AD  
   - allows multi-region replication  
   - requires direct-connect or VPN connection to on-premise AD  
   - supports three ways of trust:  
      - AWS => On-premise AD  
      - AWS <= On-premise AD  
      - AWS <=> On-premise AD  
   - important: there is no replication or synchronization between AWS Managed Microsoft AD and on-premise AD. Users are kept on each side separately. They can just use each other to check credentials. One way to keep users in cloud is to run EC2-instance, run Microsoft AD on that instance, turn on replication from on-prise AD and then set up trust connections between AWS Managed Microsoft AD and that EC2-instance's Microsoft AD.    
 - **AD Connector** - allows to connect AWS resources to on-premise AD (proxy to local AD)  
   - no-caching capability  
   - requires VPN or Direct Connect connection to on-premise AD  
   - manages users solely on-premise  
 - **Simple AD** - AD compatible directory on AWS (Samba 4), can not be joined to on-premise AD  
   - an inexpensive AD-compatible service with common directory features  
   - supports joining EC2 instances, manage users and groups  
   - Does not support MFA, RDS SQL, SSO  
   - small: 500 users, 30 groups, 20 GB storage
   - large: 5000 users, 300 groups, 50 GB storage
   - can not be joined to on-premise AD


### AWS Organizations
provides:
 - consolidated billing (one bill for all accounts + volume discounts)  
 - centralized management of accounts  
 - provides service control policies (SCPs)  
 - reserved instances may be reused by different accounts. Also management account can turn off reserved instances discount and Saving Plans discount sharing between accounts  

**How to migrate account from one AWS Organization to another?**
1. Remove account from the first organization
2. Send an invite to the second organization
3. Accept the invite in the new organization

**Service Control Policies (SCP)** allows to:  
 - define allowlist or blocklist IAM actions  
 - applied at the OU or Account level  
 - Does not apply to the management account  
 - applied to all users and roles in the account, including root user  
 - scp does not affect service-linked roles  
 - SCP must have explicit allow  

*AWS AI services opt-out policies* allows you to opt out of storing and sharing content for quality improvement for AI-services (like Rekognition, Transcribe, Polly, etc.) With SCP you can define opt-out policy for all accounts in the organization.  


### AWS IAM Identity Center
 - one-login for all AWS accounts and resources (Business cloud applications like Saleforce, Office 365, etc.), including AWS Management Console and Windows-based EC2-instances  
(successor to AWS SSO)  
You can leverage IAM Identity Center to use SSO for login to many AWS accounts  
*Persmission Set allows to define permissions for users and groups*

### AWS Control Tower
AWS Control Tower is a service that provides the easiest way to set up and govern a new, secure, multi-account AWS environment. It is designed for organizations looking to create and manage AWS environments at scale. It automates the process of setting up a new AWS environment that is secure, well-architected, and ready to use.

Here are some key features of AWS Control Tower:
 - **Account Factory**: This feature automates the process of provisioning new accounts in your AWS environment. With a few clicks, you can create a new account that is pre-configured with your organization's best-practice security baseline.

 - **Guardrails**: These are pre-packaged governance rules for security, operations, and compliance, which can be easily enabled or disabled. Guardrails help enforce your policies by providing ongoing governance for your overall AWS environment.
 There are two types of guardrails in AWS Control Tower:
    - preventive guardrails, which help prevent non-compliant actions from occurring  
    - detective guardrails, which help detect non-compliant resources or configurations  
Also there are two levels of guardrails:
    - *mandatory* guardrails, which can't be disabled. This level enforces rules that are fundamental to operating in a secure and compliant manner, such as disallowing changes to certain IAM roles or restricting access to certain services. These guardrails are intended to prevent users from performing actions that could harm your AWS environment.  
    - *strongly recommended* guardrails, which can be disabled. This level is optional, but AWS recommends enabling it to enhance the security and compliance of your environment. These guardrails can be enabled or disabled depending on the needs of your organization. They provide additional security by enforcing rules like enabling encryption for certain services or preventing public access to specific resources.
    - elective - commonly used by enterprises. Examples: "disallowing public access to S3 buckets" or "disallow delete actions without MFA in S3"  

 - **Environment Dashboard**: AWS Control Tower provides a dashboard where you can view your overall AWS environment. The dashboard provides a snapshot of your AWS environment, showing how many guardrails are enabled, how many accounts are provisioned, and the compliance status of those accounts.

 - **Blueprints**: AWS Control Tower uses blueprints to design your landing zone. The landing zone is a well-architected, multi-account AWS environment that is based on AWS best practices. Blueprints include AWS Organizations, AWS Identity and Access Management (IAM), AWS Single Sign-On (SSO), AWS CloudTrail, and Amazon Simple Storage Service (S3) bucket for log archive.

 - **Lifecycle events**: AWS Control Tower publishes lifecycle events when it updates your landing zone and when you use Account Factory to create new accounts. Lifecycle events include updating a landing zone, creating a new account, and updating an existing account.

AWS Control Tower simplifies the process of setting up a multi-account environment and incorporating best practices, making it easier for you to focus on building your applications and services in AWS.


### AWS Resource Access Manager (RAM)
Allows to share some resources between AWS accounts within AWS Organization. As a result it helps to avoid resource duplication.
You can share:
 - *VPC subnets*. You can not share security groups and default VPCs.
Participants can manage their resources there, but they can not view, delete or modify resources of other participants.  
 - *Transit Gateway*  
 - *Route53 Resolver rules* - allows to share DNS rules between accounts  
  - *License Manager* - allows to share licenses between accounts  
 - ... (many other resources)  

### AWS CloudTrail
 - provides governance, compliance and audit for your AWS account  
  - enabled by default  
  - can be enabled/disabled per region  
  - get a history of events/API calls made within your AWS account by:  
    - console  
    - SDK  
    - CLI  
    - AWS services  
  - can put logs from all regions into one S3 bucket
**Types of CloudTrail events:**
 - Management events (changes in configuration, IAM, ...)  
 - Data events - not logged by defatuls (read/write operations on S3, Lambda, DynamoDB, etc.)  
 - Insights events - not logged by default (detect unusual activity in your account, like spikes in IAM actions, etc.)  
By default events are stored for 90 days. You can store them longer by tranfwerring them to S3 bucket.  
CloudTrail events can be sent to:  
 - CloudWatch Logs  
 - S3 bucket  
 - EventBridge  
 - SNS  

**Organizational trail** - can be established in AWS Organizations. It is a trail that applies to all AWS accounts in the organization. It can not be deleted or modified by any account in the organization. It can be used to log events for all accounts in the organization.  

### KMS
Supports:  
 - symmetric encryption (you don't get accces to keys, instead you must call KMS API to encrypt/decrypt data)  
 - asymmetric encryption (you get access to public and private keys, you can encrypt data with public key and decrypt with private key by API calls)  

**Keys types:**
 - *Customer managed Keys* (CMK) - default  
   - can be used to encrypt up to 4 KB of data  
   - possibility to rotate keys (optional - every 1 year)  
   - leverage for envelope encryption  
   - can add key policy and audit in CloudTrail  
 - *AWS Managed keys* (AWS owned)  
   - used by AWS services (S3, EBS, RDS, Redshift, etc.)  
   - automatically rotated every 1 year  
   - you can't use them for your own encryption operations  
   - view key policy and audit in CloudTrail  
 - **AWS Owned CMK**
   - created and managed by AWS, used by some AWS services to protect your resources  
   - used in multiple AWS accounts, but they are not your AWS account  
   - you can't view, use, track or audit  

**CloudHSM** - cloud hardware security module  
 - dedicated hardware  
 - you entirely manage your encryption keys (not AWS)
 - FIPS 140-2 Level 3 compliance
 - provides symmetric and asymmetric encryption
 - requires cluster spread across multiple AZs
 

**KMS key material options:**
 - *KMS (default)* - KMS creates and manages the key material for you  
 - *External* - you generate key material and import it into KMS  
 - *HSM* - you generate key material in an HSM and import it into KMS. In this case your application can use the HSM to encrypt/decrypt data, but under the hood KMS will use the same HSM to encrypt/decrypt data.  

**KMS Multi-Region keys** - can be used to encrypt data in multiple regions. Keys are automatically replicated and interchangable between regions.  
**It is not global keys, but replicated keys.**

### SSM Parameter Store
 - secure storage for configuration and secrets  
 - optional seamless encryption using KMS  
 - serverless, scalable, durable, easy SDK  
 - version tracking  
 - security through IAM  
 - integration with CloudFormation (SSM parameters can be used in CloudFormation templates as inputs)  
 - parameters can be stored in hierarchies (path-like structures)  

**There are two tiers:**
 - *standatd*  
   - can store up to 10,000 parameters  
   - maximum size of a parameter is 4 KB  
   - parameter policies are not available  
   - storage pricing - free  
 - *advanced*  
   - can store up to 100,000 parameters  
   - maximum size of a parameter is 8 KB  
   - parameter policies are available  
     - TTL for parameters  
     - advanced parameters access control (e.g send event to eventbdidge when parameter is expiring)  
   - storage pricing - $0.05 per parameter per month  

### Secret Manager
  - secure storage for secrets  
  - capability to rotate secrets every X days
  - automatic generation of secrets on rotation (uses Lambda function)  
  - native integration with RDS, Redshift, DocumentDB, ECS, CloudFormation, etc.
  - supports other databases via lambda functions
  - uses KMS under the hood for encryption and decryption (therefore, to retrive key you need to have KMS permissions)

**SSM Parameter Store vs Secret Manager**
 - both are serverless
 - both are scalable
 - secret manager is more expensive
 - secret manager has a deep integration with CloudFormation
 - paramter store allows to store not encrypted parameters (KMS encryption is optional)
 - paramter store hasn't automatic rotation of secrets (can be done manually via lambda function)

### RDS Security
 - KMS encryption at rest for underlying EBS volumes and snapshots
 - Transparent Data Encryption (TDE) for Oracle and SQL Server
 - SSL encryption for data in transit for all RDS
 - IAM authentication (for MySQL and PostgreSQL and MariaDB)
 - Authorization still within RDS (not IAM)
 - CloudTrail can not be used to track queries made within RDS

### SSL/TLS
**SNI** - Server Name Indication allows to upload multiple SSL certificates on one web server. It is supported by all modern browsers.
SNI works for:
 - ALB
 - NLB
 - CloudFront
SNI doesn't work for CLB
DNSSEC allows to protect DNS from attacks. It is supported by Route53

### ACM - Certificate Manager
Provides full control over certificates management
Provides:
 - possibility to create public certificates
 - possibility to import private certificates (you create your private CA - your application should trust that CA)
 - automatic renewal of certificates
 - regional service! You can not use certificates in other regions. You need to create certificates in each region where you want to use them.

If you want to do TLS encyption on your EC2 instances, but you don't want to manage certificates and download them to EC2 instances, you can use CloudHSM. It can be integrated with nginx or other load balancers via SSL Offloading.

### S3 Security
**Encryption at rest:**
 - SSE-S3 - encryption using keys managed by S3  
 - SSE-KMS - encryption using keys managed by KMS  
  - all API calls are logged in CloudTrail as key usages  
  - objects made public can never be read  
 - SSE-C - encryption using keys managed by you  
 - Client side encryption - encryption using keys managed by you outside of AWS  
Glacier is always encrypted at rest using AES-256 under AWS control  

S3 exposes:  
 - HTTP endpoints  
 - HTTPS endpoints  
To enforce HTTPS, use a bucket policy with aws:SecureTransport  

*S3 Access Logs*  
 - logs of all requests made to S3 bucket  
 - might take an hour to deliver  
 - might be incomplete
*S3 Event Notifications*
 - about all actions
 - delivered in second or sometimes minutes
 - deduplication included (in docs it is said that insted of two events you can get only one event)  

**Tructed Advisor**  
 - check bucker permissions  

**S3 security**  
 - user based - IAM policies  
 - resource based  
   - *bucket policies* (may cross account)  
     - grant public access  
     - force objects to be encrypted at upload  
     - grant access to another account (cross account)  
     - optional conditions  
       - source Ips, source VPC, ...  
       - cloudFront Origin Identity  
   - *Object Access control list* (ACL) - finer grain  
   - *Bucket Access Control List* (ACL) - less common  

*Pre-Signed URLS*  
 - by default valid for 3600 seconds (1 hour)  

VPC Endpoint Gateway - allows to access S3 from VPC without internet gateway or NAT gateway.  

S3 Object Lock  
 - write once, read many (WORM)  
 - block an object version deletion for a specified amount of time  
 - (requires enabling versioning)  
 - supports two retention modes:
   - gogernance - users can't overwrite or delete an object unless they have special persmissions  
   - compliance - protected object can't be overwritten or deleted by any user, including root user  
 - also you can use *legal hold*. It can be attached or detached to a bucket and it prevent objects from being deleted or modified  

Glacier Vault Lock  
 - WORM for Glacier  
 - can be used to enforce compliance - any object can never be deleted  
 - you can't change policies in future  

**S3 Access Point** - allows to simplify security management for S3.
You can create many access points for different groupd of users. Each access point will have its own policy and own DNS name.

You can set up S3 Access Point, which will provide access to two or more S3 buckets in different regions. Moreover, these buckets may have a few types of replication:
 - one to one   
 - one to many  
 - many to many  
In this case Access Point may provide access to the nearest bucket to decrease latency.

### Security

**DDOS attacks:**
 - SYN Flood attack - send many SYN requests to server, but never complete the handshake. Server will wait for ACK for each SYN request.  
 - UDP Flood attack - send many UDP requests to server. Server will try to process each request.  
 - DNS flood attack - send many DNS requests to server. Server will try to process each request.  
 - Slow Loris attack - send many HTTP requests to server, but never complete the request. Server will wait for the end of request.  

**WAF**
Protects application from web exploits (layer 7)  
Deploy on ALB, API Gateway, CloudFront, AppSync  
You can defind Web ACLs (rules for IP, http headers, http body, etc.)  
Protects from SQL injection, cross-site scripting, size constraints, Geo match, rate-based rules, etc.  

**AWS Firewall Manager**
Manages rules in all accounts of an AWS organization  
 - WAF rules  
 - AWS Shield Advanced  
 - Security Groups  
 - AWS Network Firewall  
 - Amazon Route53 Resolver DNS Firewall  
**Important**: rules are applied to new resources as there are created across all and future account in the Oranization.  

**WAF vs Firewall Manager vs Shield**  
  - *WAF* - protects from web exploits  
  - *Firewall Manager* - manages rules in all accounts of an AWS organization, accelerates configuration    
  - *Shield* - protects from DDOS attacks  

