import MyCode.Git

open Lean
open MyCode

namespace MyCodeGitPlugin

private def protocolVersion : Nat := 1
private def maxFrameBytes : Nat := 1_048_576
private def maxToolOutputBytes : Nat := 262_144
private def gitDeadlineMilliseconds : Nat := 120_000

private inductive Request where
  | initialize (id : String)
  | listTools (id : String)
  | callTool (id name : String) (arguments : Json)
  | shutdown (id : String)

private def validId (id : String) : Bool :=
  !id.isEmpty && id.utf8ByteSize ≤ 64 && id.all fun char =>
    char.isAlphanum || char == '_' || char == '-'

private def parseRequest (json : Json) : Except String Request := do
  let version ← (← json.getObjVal? "v").getNat?
  if version != protocolVersion then throw s!"unsupported protocol version {version}"
  let id ← (← json.getObjVal? "id").getStr?
  if !validId id then throw "invalid correlation id"
  let operation ← (← json.getObjVal? "op").getStr?
  let parameters ← json.getObjVal? "params"
  match operation with
  | "initialize" => pure (.initialize id)
  | "list_tools" => pure (.listTools id)
  | "shutdown" => pure (.shutdown id)
  | "call_tool" =>
    let name ← (← parameters.getObjVal? "name").getStr?
    let arguments ← parameters.getObjVal? "arguments"
    pure (.callTool id name arguments)
  | _ => throw s!"unsupported plugin operation: {operation}"

private def success (id : String) (result : Json) : Json :=
  Json.mkObj [
    ("v", protocolVersion),
    ("id", id),
    ("ok", true),
    ("result", result)
  ]

private def failure (id code message : String) : Json :=
  Json.mkObj [
    ("v", protocolVersion),
    ("id", id),
    ("ok", false),
    ("error", Json.mkObj [
      ("code", code),
      ("message", message),
      ("retryable", false)
    ])
  ]

private partial def readExact? (input : IO.FS.Stream) (length : Nat) (data := ByteArray.empty) : IO (Option ByteArray) := do
  if data.size == length then
    pure (some data)
  else
    let chunk ← input.read (length - data.size).toUSize
    if chunk.isEmpty then
      if data.isEmpty then pure none else throw (IO.userError "unexpected end of plugin frame")
    else
      readExact? input length (data ++ chunk)

private def frameLength (header : ByteArray) : Nat :=
  header[0]!.toNat * 16_777_216 +
    header[1]!.toNat * 65_536 +
    header[2]!.toNat * 256 +
    header[3]!.toNat

private def readFrame? (input : IO.FS.Stream) : IO (Option Json) := do
  let some header ← readExact? input 4 | return none
  let length := frameLength header
  if length == 0 || length > maxFrameBytes then
    throw (IO.userError s!"invalid plugin frame length {length}")
  let some payload ← readExact? input length | throw (IO.userError "unexpected end of plugin frame")
  let some text := String.fromUTF8? payload | throw (IO.userError "plugin frame was not UTF-8")
  IO.ofExcept (Json.parse text)

private def writeFrame (output : IO.FS.Stream) (json : Json) : IO Unit := do
  let payload := json.compress.toUTF8
  if payload.isEmpty || payload.size > maxFrameBytes then
    throw (IO.userError "plugin response exceeded the frame limit")
  let length := payload.size
  let header := ByteArray.empty
    |>.push (length / 16_777_216).toUInt8
    |>.push (length / 65_536 % 256).toUInt8
    |>.push (length / 256 % 256).toUInt8
    |>.push (length % 256).toUInt8
  output.write header
  output.write payload
  output.flush

private structure GitContext where
  workspace : System.FilePath
  available : Bool

private partial def readBounded (handle : IO.FS.Handle) (limit : Nat) (data := ByteArray.empty) : IO String := do
  let chunk ← handle.read 8192
  if chunk.isEmpty then
    match String.fromUTF8? data with
    | some text => pure text
    | none => throw (IO.userError "Git output was not UTF-8")
  else if data.size + chunk.size > limit then
    throw (IO.userError s!"Git output exceeded {limit} bytes")
  else
    readBounded handle limit (data ++ chunk)

private partial def waitWithDeadline {cfg : IO.Process.StdioConfig}
    (child : IO.Process.Child cfg) (remaining : Nat) : IO (Option UInt32) := do
  match ← child.tryWait with
  | some code => pure (some code)
  | none =>
    if remaining == 0 then
      child.kill
      discard child.wait
      pure none
    else
      let delay := min 50 remaining
      IO.sleep delay.toUInt32
      waitWithDeadline child (remaining - delay)

private def controlledEnvironment : Array (String × Option String) :=
  #[
    ("GIT_ALTERNATE_OBJECT_DIRECTORIES", none),
    ("GIT_CONFIG", none),
    ("GIT_CONFIG_COUNT", none),
    ("GIT_CONFIG_GLOBAL", none),
    ("GIT_CONFIG_SYSTEM", none),
    ("GIT_DIR", none),
    ("GIT_EXEC_PATH", none),
    ("GIT_EXTERNAL_DIFF", none),
    ("GIT_INDEX_FILE", none),
    ("GIT_NAMESPACE", none),
    ("GIT_OBJECT_DIRECTORY", none),
    ("GIT_SSH", none),
    ("GIT_SSH_COMMAND", none),
    ("GIT_WORK_TREE", none),
    ("GIT_TERMINAL_PROMPT", some "0"),
    ("GIT_PAGER", some "cat"),
    ("PAGER", some "cat")
  ]

private def runGit (workspace : System.FilePath) (arguments : Array String) : IO (String × UInt32) := do
  let child ← IO.Process.spawn {
    cmd := "git"
    args := #["--no-optional-locks", "-c", "core.hooksPath=/dev/null"] ++ arguments
    cwd := some workspace
    stdin := .null
    stdout := .piped
    stderr := .piped
    env := controlledEnvironment
    setsid := true
  }
  let stdoutTask ← IO.asTask (readBounded child.stdout maxToolOutputBytes) Task.Priority.dedicated
  let stderrTask ← IO.asTask (readBounded child.stderr maxToolOutputBytes) Task.Priority.dedicated
  let exitCode? ← waitWithDeadline child gitDeadlineMilliseconds
  let stdout ← IO.ofExcept (← IO.wait stdoutTask)
  let stderr ← IO.ofExcept (← IO.wait stderrTask)
  let mut output := stdout
  if !stderr.isEmpty then
    if !output.isEmpty then output := output.push '\n'
    output := output ++ "[stderr]\n" ++ stderr
  match exitCode? with
  | some exitCode =>
    if exitCode != 0 then output := output ++ s!"\n[exit status: {exitCode}]"
    pure (output, exitCode)
  | none => throw (IO.userError "Git command exceeded its deadline")

private def discoverGit (workspace : System.FilePath) : IO GitContext := do
  let workspace ← IO.FS.realPath workspace
  try
    let (root, exitCode) ← runGit workspace #["rev-parse", "--show-toplevel"]
    if exitCode == 0 then
      let root ← IO.FS.realPath (System.FilePath.mk root.trimAscii.copy)
      pure { workspace, available := root == workspace }
    else
      pure { workspace, available := false }
  catch _ => pure { workspace, available := false }

private def toolSpecification (name description : String) (operations : Array String) : Json :=
  Json.mkObj [
    ("name", name),
    ("description", description),
    ("inputSchema", Json.mkObj [
      ("type", "object"),
      ("properties", Json.mkObj [
        ("operation", Json.mkObj [("type", "string"), ("enum", Json.arr (operations.map toJson))]),
        ("arguments", Json.mkObj [
          ("type", "array"),
          ("items", Json.mkObj [("type", "string")])
        ]),
        ("command", Json.mkObj [("type", "string")])
      ]),
      ("required", Json.arr ((#["operation", "arguments"] : Array String).map toJson)),
      ("additionalProperties", false)
    ])
  ]

private def toolList (context : GitContext) : Json :=
  let tools := if context.available then
    #[
      toolSpecification "git_read" "Internal verified read-only Git effect."
        #["status", "diff", "log", "show", "rev_parse", "branch_current"],
      toolSpecification "git_write" "Internal verified mutating Git effect. Requires approval."
        #["add", "restore_staged", "commit"]
    ]
  else #[]
  Json.mkObj [("tools", Json.arr tools)]

private def argumentArray (json : Json) : Except String (Array String) := do
  let values ← json.getArr?
  values.mapM Json.getStr?

private def validPath (path : String) : Bool :=
  !path.isEmpty && !path.startsWith "-" && !path.startsWith "/" && !path.contains '\\' &&
    !(path.splitOn "/").any fun segment => segment == ".." || segment == ".git"

private def validateReadArguments (arguments : Array String) : Except String Unit := do
  unless !arguments.any (fun argument =>
    argument.toList.any (fun char => char.toNat == 0) || argument == "-c" || argument == "-C" ||
      argument == "--help" || argument == "-h" || argument == "--ext-diff" ||
      argument == "--textconv" || argument == "--no-index" || argument.startsWith "--output" ||
      argument.startsWith "--git-dir" || argument.startsWith "--work-tree" ||
      argument.startsWith "--exec-path" || argument.startsWith "--config-env") do
    throw "Git command contains an option outside the verified subset"

private def validatePaths (paths : Array String) : Except String Unit := do
  if paths.isEmpty || paths.any (!validPath ·) then
    throw "Git path must be a non-empty relative path inside the repository"

private def verifiedArguments (tool operation : String) (arguments : Array String) : Except String (Array String) := do
  match tool, operation with
  | "git_read", "status" =>
    validateReadArguments arguments
    pure ((#["status"] : Array String) ++ arguments)
  | "git_read", "diff" =>
    validateReadArguments arguments
    pure ((#["diff", "--no-ext-diff", "--no-textconv"] : Array String) ++ arguments)
  | "git_read", "log" =>
    validateReadArguments arguments
    pure ((#["log", "--no-ext-diff", "--no-textconv"] : Array String) ++ arguments)
  | "git_read", "show" =>
    validateReadArguments arguments
    pure ((#["show", "--no-ext-diff", "--no-textconv"] : Array String) ++ arguments)
  | "git_read", "rev_parse" =>
    validateReadArguments arguments
    pure ((#["rev-parse"] : Array String) ++ arguments)
  | "git_read", "branch_current" =>
    if arguments.isEmpty then pure #["branch", "--show-current"]
    else throw "branch_current does not accept arguments"
  | "git_write", "add" =>
    validatePaths arguments
    pure ((#["add", "--"] : Array String) ++ arguments)
  | "git_write", "restore_staged" =>
    validatePaths arguments
    pure ((#["restore", "--staged", "--"] : Array String) ++ arguments)
  | "git_write", "commit" =>
    if arguments.size == 1 && !(arguments[0]!).isEmpty then
      pure #["commit", "--no-verify", "-m", arguments[0]!]
    else
      throw "commit requires exactly one non-empty message"
  | _, _ => throw s!"operation {operation} is not valid for tool {tool}"

private def executeTool (context : GitContext) (name : String) (arguments : Json) : IO Json := do
  if !context.available then throw (IO.userError "the configured workspace must be the root of a Git repository")
  let operation ← IO.ofExcept <| (arguments.getObjVal? "operation").bind Json.getStr?
  let argumentJson ← IO.ofExcept <| arguments.getObjVal? "arguments"
  let values ← IO.ofExcept <| argumentArray argumentJson
  let command ← IO.ofExcept <| verifiedArguments name operation values
  let (output, _) ← runGit context.workspace command
  pure (Json.mkObj [("output", output), ("truncated", false)])

private def initializeResult : Json :=
  Json.mkObj [
    ("plugin", Json.mkObj [("name", "git"), ("version", "0.1.0")]),
    ("protocol", Json.mkObj [("minVersion", 1), ("maxVersion", 1)])
  ]

private partial def serve (context : GitContext) (initialized : Bool) : IO Unit := do
  let input ← IO.getStdin
  let output ← IO.getStdout
  let some frame ← readFrame? input | pure ()
  match parseRequest frame with
  | .error message =>
    writeFrame output (failure "invalid" "invalid_request" message)
    serve context initialized
  | .ok request =>
    let (response, nextInitialized, shutdown) ←
      match request with
      | .initialize id =>
        if initialized then pure (failure id "invalid_request" "plugin is already initialized", initialized, false)
        else pure (success id initializeResult, true, false)
      | .listTools id =>
        if initialized then pure (success id (toolList context), initialized, false)
        else pure (failure id "invalid_request" "initialize must complete before tool discovery", initialized, false)
      | .callTool id name arguments =>
        if !initialized then
          pure (failure id "invalid_request" "initialize must complete before tool execution", initialized, false)
        else
          try
            let result ← executeTool context name arguments
            pure (success id result, initialized, false)
          catch error =>
            pure (failure id "tool_failed" error.toString, initialized, false)
      | .shutdown id =>
        if initialized then pure (success id (Json.mkObj []), initialized, true)
        else pure (failure id "invalid_request" "initialize must complete before shutdown", initialized, false)
    writeFrame output response
    if shutdown then pure () else serve context nextInitialized

def main : IO Unit := do
  let context ← discoverGit (← IO.currentDir)
  serve context false

end MyCodeGitPlugin

def main : IO Unit := MyCodeGitPlugin.main
