import * as vscode from "vscode";

type Mode = "fast" | "quality";

type SolveRequest = {
  function_name: string;
  signature: string; // e.g. "(n)" or "(a, b=1) -> int"
  docstring: string;
  mode: Mode;
};

type SolveResponse = {
  best?: {
    code_body?: string;
    score?: number;
    reason?: string;
  };
  warnings?: Array<{ code?: string; message?: string }>;
  timing?: { t_total_ms?: number; t_generate_ms?: number; t_eval_ms?: number };
  request_id?: string;
};

export function activate(context: vscode.ExtensionContext) {
  const output = vscode.window.createOutputChannel("AI GPT");
  context.subscriptions.push(output);

  context.subscriptions.push(
    vscode.commands.registerCommand("ai-gpt.generateFunctionBody", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) { return; }

      const doc = editor.document;
      if (doc.languageId !== "python") {
        vscode.window.showErrorMessage("Only supports Python files.");
        return;
      }

      const cfg = vscode.workspace.getConfiguration("aiGpt");
      const apiBaseUrl = String(cfg.get("apiBaseUrl", "http://127.0.0.1:8000"));
      const timeoutMs = Number(cfg.get("requestTimeoutMs", 60000));
      const defaultMode = cfg.get<Mode>("mode", "fast");

      const MODES = ["fast", "quality"] as const;

      const picked = await vscode.window.showQuickPick([...MODES], {
        title: "Select mode",
        placeHolder: `Default: ${defaultMode}`,
      });

      const mode: Mode = (picked ?? defaultMode) as Mode;
      output.appendLine("[DBG] lang=" + editor.document.languageId);
      output.appendLine("[DBG] cursorLine=" + editor.selection.active.line);
      output.appendLine("[DBG] cursorText=" + editor.document.lineAt(editor.selection.active.line).text);
      output.show(true);


      const parsed = parsePythonFunctionAtCursor(editor, output);
      if (!parsed) {
        vscode.window.showErrorMessage(
          "Put cursor inside a Python function (def ...) with signature/docstring."
        );
        return;
      }

      const { functionName, signature, docstring, defLine, bodyRange } = parsed;

      const req: SolveRequest = {
        function_name: functionName,
        signature, // IMPORTANT: matches your backend: def {name}{signature}:
        docstring,
        mode,
      };

      const url = apiBaseUrl.replace(/\/$/, "") + "/v1/solve";

      try {
        const res = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: `Generating ${functionName} (${mode})`,
            cancellable: false,
          },
          async () => await callSolveApi(url, req, timeoutMs)
        );

        const body = res.best?.code_body ?? "";
        if (!body.trim()) {
          vscode.window.showErrorMessage("API returned empty best.code_body");
          output.appendLine(
            "[EMPTY code_body] " + JSON.stringify(res, null, 2)
          );
          output.show(true);
          return;
        }

        // Indent & insert/replace body
        const indent = detectBodyIndent(editor, defLine);
        const indentedBody = indentBlock(body, indent);

        await editor.edit((eb) => {
          eb.replace(bodyRange, indentedBody + "\n");
        });

        // UX: show score/warnings/timing
        const score = res.best?.score;
        const warnCount = res.warnings?.length ?? 0;
        const tTotal = res.timing?.t_total_ms;

        vscode.window.showInformationMessage(
          `Generated ${functionName} • score=${score ?? "?"
          } • warnings=${warnCount} • ${tTotal ? `${tTotal.toFixed(0)}ms` : ""}`
        );

        output.appendLine("=== /v1/solve ===");
        output.appendLine(`fn=${functionName} sig=${signature} mode=${mode}`);
        output.appendLine(
          `score=${score ?? "?"} reason=${res.best?.reason ?? "?"}`
        );
        if (res.warnings?.length) {
          output.appendLine("warnings:");
          for (const w of res.warnings) {
            output.appendLine(`- ${w.code ?? "WARN"}: ${w.message ?? ""}`);
          }
        }
        if (res.timing) { output.appendLine("timing: " + JSON.stringify(res.timing)); }
        output.appendLine("");
        output.show(true);
      } catch (e: any) {
        vscode.window.showErrorMessage(
          `Generate failed: ${e?.message ?? String(e)}`
        );
        output.appendLine("[ERROR] " + (e?.stack ?? String(e)));
        output.show(true);
      }
    })
  );

  // optional sanity check command (keep if you want)
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "ai-gpt-function-body-generator.helloWorld",
      () => {
        vscode.window.showInformationMessage(
          "Hello World from ai-gpt-function-body-generator!"
        );
      }
    )
  );
}

export function deactivate() { }

async function callSolveApi(
  url: string,
  payload: SolveRequest,
  timeoutMs: number
): Promise<SolveResponse> {
  const controller = new AbortController();
  
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!resp.ok) {
      const txt = await resp.text().catch(() => "");
      throw new Error(`HTTP ${resp.status}: ${txt || resp.statusText}`);
    }
    return (await resp.json()) as SolveResponse;
  } finally {
    clearTimeout(t);
  }
}

/**
 * Parse function at cursor.
 * Returns signature EXACTLY in the format backend expects: "(...)" possibly with " -> ..."
 */
function parsePythonFunctionAtCursor(
  editor: vscode.TextEditor,
  output?: vscode.OutputChannel
): null | {
  functionName: string;
  signature: string;
  docstring: string;
  defLine: number;
  bodyRange: vscode.Range;
} {
  const doc = editor.document;
  const cursorLine = editor.selection.active.line;

  // 1) find def line upwards
  let defLine = -1;
  const defRegex = /^\s*(async\s+def|def)\s+(\w+)\s*\(/;

  for (let i = cursorLine; i >= 0; i--) {
    const t = doc.lineAt(i).text;
    if (defRegex.test(t)) {
      defLine = i;
      break;
    }
  }
  if (defLine < 0) {
    output?.appendLine(`[DBG] parse: no 'def ... (' found upwards from line ${cursorLine}`);
    return null;
  }

  // debug
  output?.appendLine(`[DBG] defLine=${defLine} found. function def candidate.`);

  // 2) collect signature lines until we hit ":" ending
  const sigLines: string[] = [];
  let endLine = defLine;
  for (let i = defLine; i < doc.lineCount; i++) {
    const t = doc.lineAt(i).text;
    sigLines.push(t);
    endLine = i;
    if (t.trimEnd().endsWith(":")) { break; }
    if (i - defLine > 30) { break; } // safety
  }

  const sigBlock = sigLines.join("\n");

  // function name
  const m = sigBlock.match(/^\s*(?:async\s+def|def)\s+(\w+)\s*/m);
  if (!m) {
    output?.appendLine(`[DBG] parse: regex match for function name failed on sigBlock:\n${sigBlock}`);
    return null;
  }
  const functionName = m[1];
  output?.appendLine(`[DBG] functionName=${functionName}`);

  // Extract "(...)" + optional return annotation before ":"
  // Extract signature by stripping the "def name" prefix and trailing ":"
  const oneLineForExtract = sigBlock.replace(/\r?\n/g, " ").replace(/\s+/g, " ");
  let cleanLine = oneLineForExtract.trim();
  if (cleanLine.endsWith(":")) {
    cleanLine = cleanLine.substring(0, cleanLine.length - 1).trim();
  }

  const prefixRegex = new RegExp(
    String.raw`^\s*(?:async\s+def|def)\s+${functionName}\s*`
  );
  const signature = cleanLine.replace(prefixRegex, "").trim();
  output?.appendLine(`[DBG] signature=${signature}`);

  // 3) docstring after signature
  const { docstring, bodyStartLine } = extractDocstring(doc, endLine + 1);

  // 4) body range: from first body line to next block at indent <= defIndent
  const defIndent = doc.lineAt(defLine).firstNonWhitespaceCharacterIndex;
  const start = findFirstBodyLine(doc, bodyStartLine);
  const end = findBodyEndLine(doc, start, defIndent);

  const bodyRange = new vscode.Range(
    new vscode.Position(start, 0),
    new vscode.Position(end, 0)
  );

  return { functionName, signature, docstring, defLine, bodyRange };
}

function extractDocstring(
  doc: vscode.TextDocument,
  startLine: number
): { docstring: string; bodyStartLine: number } {
  let i = startLine;

  // skip blank/comment lines
  while (i < doc.lineCount) {
    const t = doc.lineAt(i).text.trim();
    if (t === "" || t.startsWith("#")) { i++; }
    else { break; }
  }
  if (i >= doc.lineCount) { return { docstring: "", bodyStartLine: i }; }

  const raw = doc.lineAt(i).text;
  const trimmed = raw.trimStart();
  const quote = trimmed.startsWith('"""')
    ? '"""'
    : trimmed.startsWith("'''")
      ? "'''"
      : null;
  if (!quote) { return { docstring: "", bodyStartLine: i }; }

  // single-line """..."""
  if (trimmed.indexOf(quote) !== trimmed.lastIndexOf(quote)) {
    const content = trimmed.slice(3, trimmed.lastIndexOf(quote));
    return { docstring: content, bodyStartLine: i + 1 };
  }

  // multi-line
  const parts: string[] = [];
  parts.push(trimmed.slice(3));
  i++;

  while (i < doc.lineCount) {
    const t = doc.lineAt(i).text;
    const idx = t.indexOf(quote);
    if (idx >= 0) {
      parts.push(t.slice(0, idx));
      i++;
      break;
    }
    parts.push(t);
    i++;
    if (parts.length > 200) { break; }
  }

  return { docstring: parts.join("\n").trim(), bodyStartLine: i };
}

function findFirstBodyLine(doc: vscode.TextDocument, line: number): number {
  let i = line;
  while (i < doc.lineCount) {
    if (doc.lineAt(i).text.trim() !== "") {
      return i;
    }
    i++;
  }
  return Math.min(i, doc.lineCount);
}

function findBodyEndLine(
  doc: vscode.TextDocument,
  startLine: number,
  defIndent: number
): number {
  // body ends when we encounter a non-empty line with indent <= defIndent
  for (let i = startLine + 1; i < doc.lineCount; i++) {
    const line = doc.lineAt(i);
    const trimmed = line.text.trim();

    // ignore empty lines
    if (trimmed === "") {
      continue;
    }

    const indent = line.firstNonWhitespaceCharacterIndex;

    // Any dedent ends the function body
    if (indent <= defIndent) {
      return i;
    }
  }
  return doc.lineCount;
} // ✅ CỰC QUAN TRỌNG: phải có dấu } kết thúc hàm ở đây

function detectBodyIndent(editor: vscode.TextEditor, defLine: number): string {
  const doc = editor.document;
  const defIndent = doc.lineAt(defLine).text.match(/^\s*/)?.[0] ?? "";
  const tabSize = Number(editor.options.tabSize ?? 4);

  if (defIndent.includes("\t")) {
    return defIndent + "\t";
  }
  return defIndent + " ".repeat(tabSize);
}

function indentBlock(block: string, indent: string): string {
  const lines = block.split(/\r?\n/);
  return lines.map((line) => {
    if (!line.trim()) { return ""; }
    return indent + line;
  }).join("\n");
}
