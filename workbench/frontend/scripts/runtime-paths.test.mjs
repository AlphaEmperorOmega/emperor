import assert from "node:assert/strict";
import { test } from "node:test";
import {
  nextCli,
  pythonCommand,
  virtualenvPython,
} from "./runtime-paths.mjs";

test("virtualenv Python follows the native platform layout", () => {
  assert.match(virtualenvPython("project", "linux"), /torchenv[\\/]bin[\\/]python$/);
  assert.match(
    virtualenvPython("project", "win32"),
    /torchenv[\\/]Scripts[\\/]python\.exe$/,
  );
  assert.equal(pythonCommand("win32"), "python.exe");
});

test("Next resolves to its JavaScript CLI instead of a platform shim", () => {
  assert.match(nextCli("frontend"), /next[\\/]dist[\\/]bin[\\/]next$/);
});
