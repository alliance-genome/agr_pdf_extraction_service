import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import { aws_iam as iam, aws_sns as sns, aws_sns_subscriptions as subscriptions, aws_ssm as ssm } from "aws-cdk-lib";

export class OomAlertsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const projectName = new cdk.CfnParameter(this, "ProjectName", {
      type: "String",
      default: "pdfx",
      description: "Project short name",
    });

    const environmentName = new cdk.CfnParameter(this, "EnvironmentName", {
      type: "String",
      default: "dev",
      description: "Environment name (dev/staging/prod)",
    });

    const alertEmail = new cdk.CfnParameter(this, "AlertEmail", {
      type: "String",
      description: "Email address for SNS alert subscription",
    });

    const alertSsmPrefix = new cdk.CfnParameter(this, "AlertSsmPrefix", {
      type: "String",
      default: "/pdfx/alerts",
      description: "SSM path prefix for alert config parameters",
    });

    const instanceRoleName = new cdk.CfnParameter(this, "InstanceRoleName", {
      type: "String",
      default: "",
      description: "Optional existing EC2 instance role name to grant SNS publish + SSM read",
    });

    const hasInstanceRole = new cdk.CfnCondition(this, "HasInstanceRole", {
      expression: cdk.Fn.conditionNot(cdk.Fn.conditionEquals(instanceRoleName.valueAsString, "")),
    });

    const topic = new sns.Topic(this, "OomAlertTopic", {
      topicName: cdk.Fn.join("-", [projectName.valueAsString, environmentName.valueAsString, "oom-alerts"]),
      displayName: cdk.Fn.join("-", [projectName.valueAsString, environmentName.valueAsString, "OOM Alerts"]),
    });

    cdk.Tags.of(topic).add("project", projectName.valueAsString);
    cdk.Tags.of(topic).add("environment", environmentName.valueAsString);

    topic.addSubscription(new subscriptions.EmailSubscription(alertEmail.valueAsString));

    new ssm.CfnParameter(this, "OomAlertTopicArnParameter", {
      name: cdk.Fn.join("", [alertSsmPrefix.valueAsString, "/sns_topic_arn"]),
      type: "String",
      tier: "Standard",
      value: topic.topicArn,
      description: "SNS topic ARN for host-side OOM alert watcher",
    });

    new ssm.CfnParameter(this, "OomAlertEmailParameter", {
      name: cdk.Fn.join("", [alertSsmPrefix.valueAsString, "/email"]),
      type: "String",
      tier: "Standard",
      value: alertEmail.valueAsString,
      description: "Human contact email for OOM alerts",
    });

    new ssm.CfnParameter(this, "OomWatcherRegionParameter", {
      name: cdk.Fn.join("", [alertSsmPrefix.valueAsString, "/region"]),
      type: "String",
      tier: "Standard",
      value: cdk.Stack.of(this).region,
      description: "Region used by host-side OOM alert watcher",
    });

    new ssm.CfnParameter(this, "OomWatcherPollSecondsParameter", {
      name: cdk.Fn.join("", [alertSsmPrefix.valueAsString, "/dedupe_window_seconds"]),
      type: "String",
      tier: "Standard",
      value: "300",
      description: "Dedupe window for repeated OOM alerts from same source/key",
    });

    const ssmPrefixWildcardArn = cdk.Fn.sub(
      "arn:${AWS::Partition}:ssm:${AWS::Region}:${AWS::AccountId}:parameter${Prefix}/*",
      { Prefix: alertSsmPrefix.valueAsString },
    );

    const inlinePolicy = new iam.CfnPolicy(this, "OomWatcherInlinePolicy", {
      policyName: cdk.Fn.join("-", [projectName.valueAsString, environmentName.valueAsString, "oom-watcher"]),
      roles: [instanceRoleName.valueAsString],
      policyDocument: {
        Version: "2012-10-17",
        Statement: [
          {
            Sid: "PublishOomAlerts",
            Effect: "Allow",
            Action: ["sns:Publish"],
            Resource: topic.topicArn,
          },
          {
            Sid: "ReadOomWatcherConfig",
            Effect: "Allow",
            Action: ["ssm:GetParameter", "ssm:GetParameters"],
            Resource: [ssmPrefixWildcardArn],
          },
        ],
      },
    });
    inlinePolicy.cfnOptions.condition = hasInstanceRole;

    new cdk.CfnOutput(this, "TopicArn", {
      description: "SNS topic ARN for OOM alerts",
      value: topic.topicArn,
    });

    new cdk.CfnOutput(this, "TopicSsmParameterName", {
      description: "SSM parameter containing SNS topic ARN",
      value: cdk.Fn.join("", [alertSsmPrefix.valueAsString, "/sns_topic_arn"]),
    });

    new cdk.CfnOutput(this, "ConfiguredAlertEmail", {
      description: "Configured alert email endpoint (requires SNS confirmation)",
      value: alertEmail.valueAsString,
    });
  }
}
