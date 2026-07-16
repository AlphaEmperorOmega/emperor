import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import ts from "typescript";

export function readTypeScriptSource(relativePath: string) {
  const filePath = resolve(process.cwd(), relativePath);
  const sourceText = readFileSync(filePath, "utf8");
  return ts.createSourceFile(
    filePath,
    sourceText,
    ts.ScriptTarget.Latest,
    true,
    relativePath.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
}

export function findNodes<TNode extends ts.Node>(
  root: ts.Node,
  predicate: (node: ts.Node) => node is TNode,
) {
  const matches: TNode[] = [];
  function visit(node: ts.Node) {
    if (predicate(node)) {
      matches.push(node);
    }
    ts.forEachChild(node, visit);
  }
  visit(root);
  return matches;
}

export function staticImportSources(sourceFile: ts.SourceFile) {
  return sourceFile.statements
    .filter(ts.isImportDeclaration)
    .map((declaration) => declaration.moduleSpecifier)
    .filter(ts.isStringLiteralLike)
    .map((specifier) => specifier.text);
}

export function dynamicImportSources(sourceFile: ts.SourceFile) {
  return findNodes(
    sourceFile,
    (node): node is ts.CallExpression =>
      ts.isCallExpression(node) &&
      node.expression.kind === ts.SyntaxKind.ImportKeyword,
  )
    .map((call) => call.arguments[0])
    .filter(
      (argument): argument is ts.StringLiteralLike =>
        argument !== undefined && ts.isStringLiteralLike(argument),
    )
    .map((argument) => argument.text);
}

export function identifierCount(sourceFile: ts.SourceFile, name: string) {
  return findNodes(
    sourceFile,
    (node): node is ts.Identifier =>
      ts.isIdentifier(node) && node.text === name,
  ).length;
}

export function variableDeclaration(
  sourceFile: ts.SourceFile,
  name: string,
) {
  return findNodes(
    sourceFile,
    (node): node is ts.VariableDeclaration =>
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === name,
  )[0];
}

function jsxTagName(
  node: ts.JsxElement | ts.JsxSelfClosingElement,
) {
  return ts.isJsxElement(node)
    ? node.openingElement.tagName.getText()
    : node.tagName.getText();
}

export function jsxElementNames(root: ts.Node | undefined) {
  if (!root) {
    return [];
  }
  return findNodes(
    root,
    (node): node is ts.JsxElement | ts.JsxSelfClosingElement =>
      ts.isJsxElement(node) || ts.isJsxSelfClosingElement(node),
  ).map(jsxTagName);
}

export function hasJsxExpressionIdentifier(
  root: ts.Node | undefined,
  name: string,
) {
  if (!root) {
    return false;
  }
  return findNodes(
    root,
    (node): node is ts.JsxExpression =>
      ts.isJsxExpression(node) &&
      node.expression !== undefined &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === name,
  ).length > 0;
}

export function typeLiteralPropertyNames(
  sourceFile: ts.SourceFile,
  typeAliasName: string,
) {
  const alias = sourceFile.statements.find(
    (statement): statement is ts.TypeAliasDeclaration =>
      ts.isTypeAliasDeclaration(statement) &&
      statement.name.text === typeAliasName,
  );
  if (!alias) {
    return [];
  }
  const typeNode =
    ts.isTypeReferenceNode(alias.type) &&
    ts.isIdentifier(alias.type.typeName) &&
    alias.type.typeName.text === "Readonly" &&
    alias.type.typeArguments?.length === 1
      ? alias.type.typeArguments[0]
      : alias.type;
  if (!typeNode || !ts.isTypeLiteralNode(typeNode)) {
    return [];
  }
  return typeNode.members.flatMap((member) => {
    if (
      !ts.isPropertySignature(member) ||
      !member.name ||
      (!ts.isIdentifier(member.name) &&
        !ts.isStringLiteralLike(member.name))
    ) {
      return [];
    }
    return [member.name.text];
  });
}

export function useStateSetterNames(sourceFile: ts.SourceFile) {
  return findNodes(
    sourceFile,
    (node): node is ts.VariableDeclaration =>
      ts.isVariableDeclaration(node) &&
      ts.isArrayBindingPattern(node.name) &&
      node.initializer !== undefined &&
      ts.isCallExpression(node.initializer) &&
      ts.isIdentifier(node.initializer.expression) &&
      node.initializer.expression.text === "useState",
  ).flatMap((declaration) => {
    const binding = declaration.name;
    if (!ts.isArrayBindingPattern(binding)) {
      return [];
    }
    const setter = binding.elements[1];
    return setter && ts.isBindingElement(setter) && ts.isIdentifier(setter.name)
      ? [setter.name.text]
      : [];
  });
}

export function callsNamed(sourceFile: ts.SourceFile, name: string) {
  return findNodes(
    sourceFile,
    (node): node is ts.CallExpression =>
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === name,
  );
}
