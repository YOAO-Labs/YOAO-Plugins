export const PROTOCOL_VERSION = "yoao-format-adapter/v1";
export const FORMAT_ID = "labelme-json";
export const ADAPTER_VERSION = "1.0.0";
export const MANIFEST_VERSION = "1";

export const SETUP_DOCUMENT = Object.freeze({
  protocol_version: PROTOCOL_VERSION,
  format_id: FORMAT_ID,
  adapter_version: ADAPTER_VERSION,
  manifest_version: MANIFEST_VERSION,
  display_name: "LabelMe JSON",
  description:
    "Zero-dependency Node.js LabelMe JSON rectangle and polygon format adapter.",
  homepage: "https://github.com/yoao-ai/yoao-plugins",
  media_types: ["image"],
  capabilities: ["import", "export"],
  option_schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      ignore_unsupported_shapes: {
        type: "boolean",
        default: true,
        description: "Skip shape types other than rectangle and polygon.",
      },
    },
  },
});

function mapLabel(raw, mapping) {
  const labels =
    mapping && typeof mapping === "object" && mapping.labels && typeof mapping.labels === "object"
      ? mapping.labels
      : mapping && typeof mapping === "object"
        ? mapping
        : {};
  const mapped = Object.prototype.hasOwnProperty.call(labels, raw) ? labels[raw] : raw;
  return String(mapped).trim();
}

function requireFiniteNumber(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function clamp01(value, label) {
  return Math.max(0, Math.min(1, requireFiniteNumber(value, label)));
}

function parseSingleDoc(doc, mapping, ignoreUnsupported) {
  if (!doc || typeof doc !== "object" || Array.isArray(doc)) {
    throw new Error("LabelMe JSON item must be an object");
  }
  const mediaPath = String(doc.imagePath || "").trim();
  if (!mediaPath) {
    throw new Error("LabelMe JSON requires imagePath");
  }
  const width = doc.imageWidth;
  const height = doc.imageHeight;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error("LabelMe JSON imageWidth and imageHeight must be positive integers");
  }
  const shapes = Array.isArray(doc.shapes) ? doc.shapes : [];
  const annotations = [];
  const warnings = [];

  for (let idx = 0; idx < shapes.length; idx += 1) {
    const shape = shapes[idx];
    if (!shape || typeof shape !== "object") {
      continue;
    }
    const rawLabel = String(shape.label || "").trim();
    if (!rawLabel) {
      throw new Error(`LabelMe shape at index ${idx} requires label`);
    }
    const className = mapLabel(rawLabel, mapping);
    const shapeType = String(shape.shape_type || "polygon").toLowerCase();
    const points = Array.isArray(shape.points) ? shape.points : [];
    const sourceId = shape.group_id !== undefined && shape.group_id !== null
      ? `${idx}:${shape.group_id}`
      : String(idx);

    if (shapeType === "rectangle") {
      if (points.length !== 2) {
        throw new Error(`LabelMe rectangle at index ${idx} requires exactly 2 points`);
      }
      const [p1, p2] = points;
      if (!Array.isArray(p1) || !Array.isArray(p2) || p1.length < 2 || p2.length < 2) {
        throw new Error(`LabelMe rectangle at index ${idx} has invalid coordinates`);
      }
      const x1 = clamp01(requireFiniteNumber(p1[0], `rectangle[${idx}].p1.x`) / width, `rectangle[${idx}].x1`);
      const y1 = clamp01(requireFiniteNumber(p1[1], `rectangle[${idx}].p1.y`) / height, `rectangle[${idx}].y1`);
      const x2 = clamp01(requireFiniteNumber(p2[0], `rectangle[${idx}].p2.x`) / width, `rectangle[${idx}].x2`);
      const y2 = clamp01(requireFiniteNumber(p2[1], `rectangle[${idx}].p2.y`) / height, `rectangle[${idx}].y2`);
      const x = Math.min(x1, x2);
      const y = Math.min(y1, y2);
      const w = Math.abs(x2 - x1);
      const h = Math.abs(y2 - y1);
      if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
        throw new Error(`LabelMe rectangle at index ${idx} has zero area`);
      }
      annotations.push({
        class_name: className,
        tool_unit_id: "bbox",
        geometry: { type: "bbox", x, y, w, h },
        source_id: sourceId,
        attributes: {},
      });
    } else if (shapeType === "polygon") {
      if (points.length < 3) {
        throw new Error(`LabelMe polygon at index ${idx} requires at least 3 points`);
      }
      const normalized = points.map((pt, ptIdx) => {
        if (!Array.isArray(pt) || pt.length < 2) {
          throw new Error(`LabelMe polygon at index ${idx} has invalid point`);
        }
        return [
          clamp01(requireFiniteNumber(pt[0], `polygon[${idx}].p[${ptIdx}].x`) / width, `polygon[${idx}].x`),
          clamp01(requireFiniteNumber(pt[1], `polygon[${idx}].p[${ptIdx}].y`) / height, `polygon[${idx}].y`),
        ];
      });
      annotations.push({
        class_name: className,
        tool_unit_id: "region",
        geometry: { type: "polygon", points: normalized },
        source_id: sourceId,
        attributes: {},
      });
    } else if (ignoreUnsupported) {
      warnings.push({ code: "unsupported_geometry", shape_type: shapeType, index: idx });
    } else {
      throw new Error(`Unsupported LabelMe shape_type: ${shapeType}`);
    }
  }

  return {
    media_path: mediaPath,
    width,
    height,
    losses: [],
    warnings,
    annotations,
    mask_objects: {},
  };
}

export function parseLabelMeImport({ content_base64, mapping = {}, options = {} }) {
  if (typeof content_base64 !== "string" || !content_base64) {
    throw new Error("content_base64 is required");
  }
  const rawText = Buffer.from(content_base64, "base64").toString("utf8");
  let parsed;
  try {
    parsed = JSON.parse(rawText);
  } catch {
    throw new Error("LabelMe payload is not valid JSON");
  }
  const ignoreUnsupported = options.ignore_unsupported_shapes !== false;
  const docs = Array.isArray(parsed) ? parsed : [parsed];
  const items = docs.map((doc) => parseSingleDoc(doc, mapping, ignoreUnsupported));
  return {
    protocol_version: PROTOCOL_VERSION,
    items,
  };
}

function safeStem(filePath, index) {
  const leaf = String(filePath || `task_${index}`)
    .replaceAll("\\", "/")
    .split("/")
    .filter(Boolean)
    .pop() || `task_${index}`;
  const stem = leaf.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_") || `task_${index}`;
  return stem;
}

export function renderLabelMeExport({ tasks = [] }) {
  const files = [];
  const used = new Set();
  for (let idx = 0; idx < tasks.length; idx += 1) {
    const task = tasks[idx] || {};
    const width =
      typeof task.width === "number" && Number.isFinite(task.width) && task.width > 0
        ? task.width
        : 1000;
    const height =
      typeof task.height === "number" && Number.isFinite(task.height) && task.height > 0
        ? task.height
        : 1000;
    const shapes = [];
    for (const ann of Array.isArray(task.annotations) ? task.annotations : []) {
      const geom = ann && typeof ann.geometry === "object" ? ann.geometry : {};
      if (geom.type === "bbox") {
        const gx = requireFiniteNumber(geom.x, "bbox.x");
        const gy = requireFiniteNumber(geom.y, "bbox.y");
        const gw = requireFiniteNumber(geom.w, "bbox.w");
        const gh = requireFiniteNumber(geom.h, "bbox.h");
        const x1 = gx * width;
        const y1 = gy * height;
        const x2 = (gx + gw) * width;
        const y2 = (gy + gh) * height;
        shapes.push({
          label: String(ann.class_name || ""),
          points: [[x1, y1], [x2, y2]],
          group_id: null,
          shape_type: "rectangle",
          flags: {},
        });
      } else if (geom.type === "polygon" && Array.isArray(geom.points)) {
        shapes.push({
          label: String(ann.class_name || ""),
          points: geom.points.map(([px, py]) => [
            requireFiniteNumber(px, "polygon.x") * width,
            requireFiniteNumber(py, "polygon.y") * height,
          ]),
          group_id: null,
          shape_type: "polygon",
          flags: {},
        });
      }
    }
    const base = safeStem(task.file_path, idx);
    let relPath = `${base}.json`;
    let suffix = 1;
    while (used.has(relPath)) {
      relPath = `${base}_${suffix}.json`;
      suffix += 1;
    }
    used.add(relPath);
    const labelmeDoc = {
      version: "5.0.0",
      flags: {},
      shapes,
      imagePath: String(task.file_path || `${base}.jpg`),
      imageData: null,
      imageHeight: height,
      imageWidth: width,
    };
    files.push({
      path: relPath,
      content_base64: Buffer.from(JSON.stringify(labelmeDoc, null, 2), "utf8").toString("base64"),
    });
  }
  return {
    protocol_version: PROTOCOL_VERSION,
    files,
  };
}
