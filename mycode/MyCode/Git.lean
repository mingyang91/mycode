import MyCode.Protocol

open Lean

namespace MyCode

public inductive GitCapability where
  | inspect
  | indexWrite
  | historyWrite
  deriving BEq, DecidableEq, Repr, ToJson, FromJson

public inductive GitIntent where
  | status (args : Array String)
  | diff (args : Array String)
  | log (args : Array String)
  | show (args : Array String)
  | revParse (args : Array String)
  | branchCurrent
  | add (paths : Array String)
  | restoreStaged (paths : Array String)
  | commit (message : String)
  deriving BEq, Repr, ToJson, FromJson

public def GitIntent.requiredCapabilities : GitIntent → List GitCapability
  | .status _ | .diff _ | .log _ | .show _ | .revParse _ | .branchCurrent => [.inspect]
  | .add _ | .restoreStaged _ => [.indexWrite]
  | .commit _ => [.historyWrite]

public def GitIntent.isReadOnly : GitIntent → Bool
  | .status _ | .diff _ | .log _ | .show _ | .revParse _ | .branchCurrent => true
  | .add _ | .restoreStaged _ | .commit _ => false

public def GitIntent.internalToolName (intent : GitIntent) : String :=
  if intent.isReadOnly then "git_read" else "git_write"

public def GitIntent.operation : GitIntent → String
  | .status _ => "status"
  | .diff _ => "diff"
  | .log _ => "log"
  | .show _ => "show"
  | .revParse _ => "rev_parse"
  | .branchCurrent => "branch_current"
  | .add _ => "add"
  | .restoreStaged _ => "restore_staged"
  | .commit _ => "commit"

public def GitIntent.arguments : GitIntent → Array String
  | .status args | .diff args | .log args | .show args | .revParse args => args
  | .branchCurrent => #[]
  | .add paths | .restoreStaged paths => paths
  | .commit message => #[message]

public def GitIntent.executorArguments (intent : GitIntent) (command : String) : Json :=
  Json.mkObj [
    ("operation", toJson intent.operation),
    ("arguments", toJson intent.arguments),
    ("command", toJson command)
  ]

private inductive QuoteMode where
  | plain
  | single
  | double
  deriving BEq

private def finishWord (started : Bool) (current : String) (words : Array String) : Array String × String × Bool :=
  if started then (words.push current, "", false) else (words, current, started)

private def tokenizeLoop : List Char → QuoteMode → Bool → Bool → String → Array String → Except String (Array String)
  | [], .plain, false, started, current, words =>
    let (words, _, _) := finishWord started current words
    pure words
  | [], _, _, _, _, _ => throw "unterminated shell quote or escape"
  | char :: rest, mode, escaped, started, current, words =>
    if escaped then
      tokenizeLoop rest mode false true (current.push char) words
    else
      match mode with
      | .single =>
        if char == '\'' then tokenizeLoop rest .plain false true current words
        else tokenizeLoop rest .single false true (current.push char) words
      | .double =>
        if char == '"' then tokenizeLoop rest .plain false true current words
        else if char == '\\' then tokenizeLoop rest .double true true current words
        else if char == '$' || char == '`' then throw "dynamic shell expressions are not supported"
        else tokenizeLoop rest .double false true (current.push char) words
      | .plain =>
        if char == '\n' || char == '\r' then throw "multiple shell commands are not supported"
        else if char.isWhitespace then
          let (words, current, started) := finishWord started current words
          tokenizeLoop rest .plain false started current words
        else if char == '\'' then tokenizeLoop rest .single false true current words
        else if char == '"' then tokenizeLoop rest .double false true current words
        else if char == '\\' then tokenizeLoop rest .plain true true current words
        else if "|&;<>$`(){}".contains char then throw "shell operators and dynamic expressions are not supported"
        else tokenizeLoop rest .plain false true (current.push char) words

public def tokenizeShellCommand (command : String) : Except String (Array String) :=
  tokenizeLoop command.toList .plain false false "" #[]

private def isPathArgument (value : String) : Bool :=
  !value.isEmpty && !value.startsWith "-" && value != ".git" &&
    !value.startsWith ".git/" && !value.startsWith "/" &&
    !(value.splitOn "/").any (· == "..")

private def parsePaths (args : List String) : Option (Array String) :=
  let paths := match args with
    | "--" :: rest => rest
    | rest => rest
  if !paths.isEmpty && paths.all isPathArgument then some paths.toArray else none

private def hasForbiddenReadOption (args : List String) : Bool :=
  args.any fun arg =>
    arg == "--help" || arg == "-h" || arg == "--ext-diff" ||
    arg == "--textconv" || arg == "--no-index" || arg.startsWith "--git-dir" ||
    arg.startsWith "--work-tree" || arg == "-c" || arg == "-C"

private def parseGitWords (words : List String) : Option GitIntent :=
  let words := match words with
    | "git" :: "--no-pager" :: rest => "git" :: rest
    | other => other
  match words with
  | "git" :: "status" :: args =>
    if hasForbiddenReadOption args then none else some (.status args.toArray)
  | "git" :: "diff" :: args =>
    if hasForbiddenReadOption args then none else some (.diff args.toArray)
  | "git" :: "log" :: args =>
    if hasForbiddenReadOption args then none else some (.log args.toArray)
  | "git" :: "show" :: args =>
    if hasForbiddenReadOption args then none else some (.show args.toArray)
  | "git" :: "rev-parse" :: args =>
    if hasForbiddenReadOption args then none else some (.revParse args.toArray)
  | ["git", "branch"] | ["git", "branch", "--show-current"] => some .branchCurrent
  | "git" :: "add" :: args => .add <$> parsePaths args
  | "git" :: "restore" :: "--staged" :: args => .restoreStaged <$> parsePaths args
  | ["git", "commit", "-m", message] | ["git", "commit", "--message", message] =>
    if message.isEmpty then none else some (.commit message)
  | _ => none

public def parseGitCommand (command : String) : Option GitIntent :=
  match tokenizeShellCommand command with
  | .ok words => parseGitWords words.toList
  | .error _ => none

private def commandArgument? (arguments : Json) : Option String :=
  match arguments.getObjVal? "command" with
  | .ok (.str command) => some command
  | _ => none

private def isAutoLsArgument (value : String) : Bool :=
  value == "." ||
    (value.startsWith "-" && !value.contains "/" && !value.contains ".." &&
      !value.contains "H" && !value.contains "L" && !value.contains "dereference")

private def autoReadCommandWords : List String → Bool
  | ["pwd"] => true
  | "ls" :: args => args.all isAutoLsArgument
  | _ => false

public def ToolCall.autoPermissionAllowed (call : ToolCall) : Bool :=
  if call.name != "bash" then false
  else
    match commandArgument? call.arguments with
    | none => false
    | some command =>
      match tokenizeShellCommand command with
      | .ok words => autoReadCommandWords words.toList
      | .error _ => false

public def lowerGitToolCall? (call : ToolCall) : Option ToolCall := do
  if call.name != "bash" then none else pure ()
  let command ← commandArgument? call.arguments
  let intent ← parseGitCommand command
  pure {
    call with
    name := intent.internalToolName
    arguments := intent.executorArguments command
  }

public def lowerGitToolCalls (calls : Array ToolCall) : Array ToolCall :=
  calls.map fun call => (lowerGitToolCall? call).getD call

public theorem lower_git_tool_calls_preserves_count (calls : Array ToolCall) :
    (lowerGitToolCalls calls).size = calls.size := by
  simp [lowerGitToolCalls]

public structure AbstractGitState where
  headGeneration : Nat := 0
  objectGeneration : Nat := 0
  indexGeneration : Nat := 0
  worktreeGeneration : Nat := 0
  remoteGeneration : Nat := 0
  deriving BEq, Repr

public def AbstractGitState.Valid (state : AbstractGitState) : Prop :=
  state.headGeneration ≤ state.objectGeneration

public def GitIntent.step (intent : GitIntent) (state : AbstractGitState) : AbstractGitState :=
  match intent with
  | .status _ | .diff _ | .log _ | .show _ | .revParse _ | .branchCurrent => state
  | .add _ | .restoreStaged _ => { state with indexGeneration := state.indexGeneration + 1 }
  | .commit _ =>
    let generation := state.objectGeneration + 1
    { state with headGeneration := generation, objectGeneration := generation }

public theorem GitIntent.step_preserves_valid (intent : GitIntent) (state : AbstractGitState)
    (valid : state.Valid) : (intent.step state).Valid := by
  cases intent <;> simp [GitIntent.step, AbstractGitState.Valid] <;> assumption

public theorem GitIntent.read_only_preserves_state (intent : GitIntent) (state : AbstractGitState)
    (readOnly : intent.isReadOnly = true) : intent.step state = state := by
  cases intent <;> simp_all [GitIntent.isReadOnly, GitIntent.step]

public theorem GitIntent.read_tool_requires_inspect (intent : GitIntent)
    (readTool : intent.internalToolName = "git_read") :
    intent.requiredCapabilities = [.inspect] := by
  cases intent <;> simp_all [GitIntent.internalToolName, GitIntent.isReadOnly,
    GitIntent.requiredCapabilities]

public theorem parsed_git_tool_is_classified (command : String) (intent : GitIntent)
    (_parsed : parseGitCommand command = some intent) :
    intent.internalToolName = "git_read" ∨ intent.internalToolName = "git_write" := by
  cases intent <;> simp [GitIntent.internalToolName, GitIntent.isReadOnly]

public structure GitExecutionResult where
  output : String
  exitCode : Int
  deriving BEq, Repr

public abbrev GitObservation :=
  GitIntent → AbstractGitState → GitExecutionResult → AbstractGitState → Prop

public def GitHandlerRefines (observed : GitObservation) : Prop :=
  ∀ intent before result after,
    observed intent before result after → after = intent.step before

public theorem git_handler_preserves_valid (observed : GitObservation)
    (refines : GitHandlerRefines observed) (intent : GitIntent)
    (before after : AbstractGitState) (result : GitExecutionResult)
    (valid : before.Valid) (execution : observed intent before result after) : after.Valid := by
  rw [refines intent before result after execution]
  exact intent.step_preserves_valid before valid

end MyCode
