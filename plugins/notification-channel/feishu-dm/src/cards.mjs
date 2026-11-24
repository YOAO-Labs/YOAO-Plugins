const EVENT_TITLES = Object.assign(Object.create(null), {
  "task.approved": {
    zh: "任务审核通过",
    en: "Task approved",
    template: "green",
  },
  "task.rejected": { zh: "任务被驳回", en: "Task rejected", template: "red" },
  "task.reopened": {
    zh: "任务已重新打开",
    en: "Task reopened",
    template: "orange",
  },
  "batch.rejected": { zh: "批次被驳回", en: "Batch rejected", template: "red" },
  "batch.review_reopened": {
    zh: "批次重新进入审核",
    en: "Batch review reopened",
    template: "orange",
  },
  "batch.admin_locked": {
    zh: "批次已被锁定",
    en: "Batch locked",
    template: "orange",
  },
  "batch.admin_unlocked": {
    zh: "批次已解锁",
    en: "Batch unlocked",
    template: "blue",
  },
  "batch.unarchived": {
    zh: "批次已取消归档",
    en: "Batch unarchived",
    template: "blue",
  },
  "feedback.reply_created": {
    zh: "反馈收到新回复",
    en: "New feedback reply",
    template: "blue",
  },
  "feedback.status_changed": {
    zh: "反馈状态变更",
    en: "Feedback status changed",
    template: "blue",
  },
  "feedback.comment_mentioned": {
    zh: "任务讨论中提到了你",
    en: "Mentioned in task discussion",
    template: "indigo",
  },
  "annotation.comment_mentioned": {
    zh: "标注评论中提到了你",
    en: "Mentioned in annotation comment",
    template: "indigo",
  },
  "bug_report.commented": {
    zh: "Bug 反馈收到新评论",
    en: "New bug report comment",
    template: "blue",
  },
  "bug_report.reopened": {
    zh: "Bug 反馈已重开",
    en: "Bug report reopened",
    template: "orange",
  },
  "bug_report.status_changed": {
    zh: "Bug 反馈状态更新",
    en: "Bug report status changed",
    template: "blue",
  },
  "failed_prediction.retry.started": {
    zh: "失败预测开始重试",
    en: "Prediction retry started",
    template: "blue",
  },
  "failed_prediction.retry.succeeded": {
    zh: "失败预测重试成功",
    en: "Prediction retry succeeded",
    template: "green",
  },
  "failed_prediction.retry.failed": {
    zh: "失败预测重试失败",
    en: "Prediction retry failed",
    template: "red",
  },
  "export.ready": { zh: "导出已完成", en: "Export ready", template: "green" },
  "export.failed": { zh: "导出失败", en: "Export failed", template: "red" },
  "job.completed": {
    zh: "后台任务已完成",
    en: "Background job completed",
    template: "green",
  },
  "job.failed": {
    zh: "后台任务失败",
    en: "Background job failed",
    template: "red",
  },
  "job.cancelled": {
    zh: "后台任务已取消",
    en: "Background job cancelled",
    template: "grey",
  },
});

export function escapeLarkMd(value) {
  return String(value).replace(/([\\`*_{}[\]()#+\-.!|>~])/g, "\\$1");
}

function isEnglish(locale) {
  return typeof locale === "string" && locale.toLowerCase().startsWith("en");
}

function resolveActionUrl(publicBaseUrl, actionUrl) {
  if (
    !publicBaseUrl ||
    typeof actionUrl !== "string" ||
    !actionUrl.startsWith("/")
  ) {
    return null;
  }
  if (
    actionUrl.startsWith("//") ||
    /[\x00-\x1f\x7f\s]/.test(actionUrl) ||
    /(?:^|\/)\.\.(?:\/|$)/.test(actionUrl)
  ) {
    return null;
  }
  return `${publicBaseUrl}${actionUrl}`;
}

function buildSummaryLine(event, data, en) {
  const parts = [];
  if (typeof data.task_display_id === "string" && data.task_display_id) {
    parts.push(
      en
        ? `Task ${escapeLarkMd(data.task_display_id)}`
        : `任务 ${escapeLarkMd(data.task_display_id)}`,
    );
  } else if (
    typeof data.batch_display_id === "string" &&
    data.batch_display_id
  ) {
    parts.push(
      en
        ? `Batch ${escapeLarkMd(data.batch_display_id)}`
        : `批次 ${escapeLarkMd(data.batch_display_id)}`,
    );
  } else if (typeof data.display_id === "string" && data.display_id) {
    parts.push(escapeLarkMd(data.display_id));
  } else if (typeof data.kind === "string" && data.kind) {
    parts.push(
      en ? `Job ${escapeLarkMd(data.kind)}` : `作业 ${escapeLarkMd(data.kind)}`,
    );
  }

  if (typeof data.project_name === "string" && data.project_name) {
    parts.push(
      en
        ? `Project ${escapeLarkMd(data.project_name)}`
        : `项目 ${escapeLarkMd(data.project_name)}`,
    );
  } else if (
    typeof data.project_display_id === "string" &&
    data.project_display_id
  ) {
    parts.push(
      en
        ? `Project ${escapeLarkMd(data.project_display_id)}`
        : `项目 ${escapeLarkMd(data.project_display_id)}`,
    );
  }

  if (typeof data.reject_reason_type === "string" && data.reject_reason_type) {
    parts.push(
      en
        ? `Reason: ${escapeLarkMd(data.reject_reason_type)}`
        : `原因分类：${escapeLarkMd(data.reject_reason_type)}`,
    );
  }
  if (typeof data.to_status === "string" && data.to_status) {
    const transition =
      typeof data.from_status === "string" && data.from_status
        ? `${escapeLarkMd(data.from_status)} → ${escapeLarkMd(data.to_status)}`
        : escapeLarkMd(data.to_status);
    parts.push(en ? `Status: ${transition}` : `状态：${transition}`);
  }
  if (typeof data.actor_name === "string" && data.actor_name) {
    parts.push(
      en
        ? `By ${escapeLarkMd(data.actor_name)}`
        : `操作人：${escapeLarkMd(data.actor_name)}`,
    );
  }

  if (parts.length === 0) {
    return en ? `Event ${escapeLarkMd(event)}` : `事件 ${escapeLarkMd(event)}`;
  }
  return parts.join(" · ");
}

export function buildNotificationCard({
  envelope,
  locale,
  publicBaseUrl = "",
}) {
  const en = isEnglish(locale);
  const meta = Object.hasOwn(EVENT_TITLES, envelope.event)
    ? EVENT_TITLES[envelope.event]
    : {
        zh: `YOAO 通知：${envelope.event}`,
        en: `YOAO notification: ${envelope.event}`,
        template: "blue",
      };
  const title = en ? meta.en : meta.zh;
  const data =
    envelope.data && typeof envelope.data === "object" ? envelope.data : {};
  const summary = buildSummaryLine(envelope.event, data, en);
  const targetUrl = resolveActionUrl(publicBaseUrl, data.action_url);

  const elements = [
    {
      tag: "div",
      text: {
        tag: "lark_md",
        content: summary,
      },
    },
    {
      tag: "note",
      elements: [
        {
          tag: "plain_text",
          content: `YOAO · ${envelope.event} · ${envelope.occurred_at}`,
        },
      ],
    },
  ];

  if (targetUrl) {
    const buttonText =
      data.target_type === "task" ||
      data.target_type === "feedback" ||
      data.target_type === "annotation_comment"
        ? en
          ? "Open Workbench"
          : "打开工作台"
        : en
          ? "View in YOAO"
          : "在 YOAO 中查看";
    elements.splice(1, 0, {
      tag: "action",
      actions: [
        {
          tag: "button",
          type: "primary",
          text: {
            tag: "plain_text",
            content: buttonText,
          },
          url: targetUrl,
        },
      ],
    });
  }

  return {
    config: { wide_screen_mode: true },
    header: {
      template: meta.template,
      title: {
        tag: "plain_text",
        content: title,
      },
    },
    elements,
  };
}

export function buildVerifyAddressCard({ locale, verificationId }) {
  const en = isEnglish(locale);
  return {
    config: { wide_screen_mode: true },
    header: {
      template: "blue",
      title: {
        tag: "plain_text",
        content: en
          ? "YOAO Feishu Notification Verification"
          : "YOAO 飞书通知地址验证",
      },
    },
    elements: [
      {
        tag: "div",
        text: {
          tag: "lark_md",
          content: en
            ? "Your Feishu account is now connected to YOAO notifications."
            : "你的飞书账号已成功连通 YOAO 外部通知通道。",
        },
      },
      {
        tag: "note",
        elements: [
          {
            tag: "plain_text",
            content: `verification_id: ${verificationId}`,
          },
        ],
      },
    ],
  };
}
