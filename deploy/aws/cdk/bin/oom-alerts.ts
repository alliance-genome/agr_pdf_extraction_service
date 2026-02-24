#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { OomAlertsStack } from "../lib/oom-alerts-stack";

const app = new cdk.App();

new OomAlertsStack(app, "OomAlertsStack", {
  description: "PDFX OOM alert infrastructure (SNS + SSM + optional role policy)",
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});
