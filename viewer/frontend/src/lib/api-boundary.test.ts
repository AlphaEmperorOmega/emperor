import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { describe, expect, it } from "vitest";

type BackendContactException = {
  relativePath: string;
  reason: string;
};

const BACKEND_CONTACT_EXCEPTIONS: BackendContactException[] = [
  {
    relativePath: "src/lib/api.ts",
    reason: "Central API client owns fetch, base URL, and backend route paths.",
  },
];

const BACKEND_ROUTE_PREFIXES = [
  "/health",
  "/capabilities",
  "/models",
  "/inspect",
  "/training",
  "/logs",
] as const;

const FORBIDDEN_ENV_NAME = "NEXT_PUBLIC_VIEWER_API_URL";
const FORBIDDEN_LOCAL_API_URL = "http://127.0.0.1:9999";
const PRODUCTION_SOURCE_DIRS = ["src", "app"] as const;
const SKIPPED_DIR_NAMES = new Set([
  ".next",
  "build",
  "coverage",
  "dist",
  "generated",
  "node_modules",
  "out",
]);

const testFilePath = fileURLToPath(import.meta.url);
const frontendRoot = path.resolve(path.dirname(testFilePath), "../..");
const allowedBackendContactFiles = new Set(
  BACKEND_CONTACT_EXCEPTIONS.map((entry) => entry.relativePath),
);

type BoundaryViolation = {
  filePath: string;
  line: number;
  column: number;
  message: string;
};

function toFrontendRelativePath(filePath: string) {
  return path.relative(frontendRoot, filePath).split(path.sep).join("/");
}

function isProductionSourceFile(filePath: string) {
  if (!/\.(ts|tsx)$/.test(filePath)) {
    return false;
  }
  if (/\.d\.ts$/.test(filePath)) {
    return false;
  }
  return !/\.(test|spec)\.(ts|tsx)$/.test(filePath);
}

function productionSourceFilesIn(directoryPath: string): string[] {
  return readdirSync(directoryPath).flatMap((entryName) => {
    const entryPath = path.join(directoryPath, entryName);
    const entryStats = statSync(entryPath);
    if (entryStats.isDirectory()) {
      if (SKIPPED_DIR_NAMES.has(entryName)) {
        return [];
      }
      return productionSourceFilesIn(entryPath);
    }
    return entryStats.isFile() && isProductionSourceFile(entryPath) ? [entryPath] : [];
  });
}

function productionSourceFiles() {
  return PRODUCTION_SOURCE_DIRS.flatMap((directoryName) => {
    const directoryPath = path.join(frontendRoot, directoryName);
    return existsSync(directoryPath) ? productionSourceFilesIn(directoryPath) : [];
  });
}

function blockedBackendRoutePrefix(value: string) {
  return BACKEND_ROUTE_PREFIXES.find(
    (prefix) =>
      value === prefix || value.startsWith(`${prefix}/`) || value.startsWith(`${prefix}?`),
  );
}

function fetchCallName(expression: ts.Expression, sourceFile: ts.SourceFile) {
  if (ts.isIdentifier(expression) && expression.text === "fetch") {
    return "fetch(...)";
  }
  if (
    ts.isPropertyAccessExpression(expression) &&
    expression.name.text === "fetch" &&
    ["globalThis", "self", "window"].includes(expression.expression.getText(sourceFile))
  ) {
    return `${expression.expression.getText(sourceFile)}.fetch(...)`;
  }
  if (
    ts.isElementAccessExpression(expression) &&
    ts.isStringLiteralLike(expression.argumentExpression) &&
    expression.argumentExpression.text === "fetch" &&
    ["globalThis", "self", "window"].includes(expression.expression.getText(sourceFile))
  ) {
    return `${expression.expression.getText(sourceFile)}["fetch"](...)`;
  }
  return null;
}

function addViolation(
  violations: BoundaryViolation[],
  sourceFile: ts.SourceFile,
  node: ts.Node,
  message: string,
) {
  const position = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
  violations.push({
    filePath: sourceFile.fileName,
    line: position.line + 1,
    column: position.character + 1,
    message,
  });
}

function scanLiteralText(
  violations: BoundaryViolation[],
  sourceFile: ts.SourceFile,
  node: ts.Node,
  value: string,
) {
  const routePrefix = blockedBackendRoutePrefix(value);
  if (routePrefix) {
    addViolation(
      violations,
      sourceFile,
      node,
      `backend route literal starts with ${routePrefix}`,
    );
  }
  if (value === FORBIDDEN_ENV_NAME) {
    addViolation(violations, sourceFile, node, `direct ${FORBIDDEN_ENV_NAME} reference`);
  }
  if (value.startsWith(FORBIDDEN_LOCAL_API_URL)) {
    addViolation(violations, sourceFile, node, `direct local API URL literal`);
  }
}

function scanFile(filePath: string) {
  const sourceText = readFileSync(filePath, "utf8");
  const scriptKind = filePath.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const sourceFile = ts.createSourceFile(
    filePath,
    sourceText,
    ts.ScriptTarget.Latest,
    true,
    scriptKind,
  );
  const violations: BoundaryViolation[] = [];

  function visit(node: ts.Node) {
    if (ts.isCallExpression(node)) {
      const callName = fetchCallName(node.expression, sourceFile);
      if (callName) {
        addViolation(violations, sourceFile, node, `direct ${callName} call`);
      }
    }
    if (ts.isStringLiteralLike(node)) {
      scanLiteralText(violations, sourceFile, node, node.text);
    }
    if (ts.isTemplateExpression(node)) {
      scanLiteralText(violations, sourceFile, node.head, node.head.text);
    }
    if (
      ts.isPropertyAccessExpression(node) &&
      node.name.text === FORBIDDEN_ENV_NAME &&
      node.expression.getText(sourceFile) === "process.env"
    ) {
      addViolation(violations, sourceFile, node, `direct ${FORBIDDEN_ENV_NAME} reference`);
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  return violations;
}

function isAllowedBackendContact(filePath: string) {
  return allowedBackendContactFiles.has(toFrontendRelativePath(filePath));
}

function formatViolation(violation: BoundaryViolation) {
  return `${toFrontendRelativePath(violation.filePath)}:${violation.line}:${
    violation.column
  } - ${violation.message}`;
}

describe("frontend API boundary", () => {
  it("keeps production backend contact centralized in src/lib/api.ts", () => {
    const files = productionSourceFiles();
    const violations = files
      .flatMap(scanFile)
      .filter((violation) => !isAllowedBackendContact(violation.filePath))
      .map(formatViolation);

    expect(files.map(toFrontendRelativePath)).toContain("src/lib/api.ts");
    expect(violations.join("\n")).toBe("");
  });
});
