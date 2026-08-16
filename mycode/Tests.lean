import MyCode.State

open Lean
open MyCode

private def assert (condition : Bool) (label : String) : IO Unit :=
  unless condition do throw (IO.userError label)

private def step (state : State) (event : Event) : IO (State × Array Effect) := do
  match transition state event with
  | .ok result => pure result
  | .error error => throw (IO.userError error)

private def call (callId name : String) : ToolCall := {
  callId
  name
  arguments := Json.mkObj []
}

private def bashCall (callId command : String) : ToolCall := {
  callId
  name := "bash"
  arguments := Json.mkObj [("command", toJson command)]
}

private def testSafeToolFlowsDirectlyToExecutor : IO Unit := do
  let (configured, _) ← step {} { kind := "configure_tools", safeTools := #["read"] }
  let (submitted, submitEffects) ← step configured { kind := "submit", text? := some "Inspect Main.lean" }
  assert (submitEffects.size == 1 && submitEffects[0]!.kind == "request_model") "submit must request a model response"
  let (awaitingTool, effects) ← step submitted {
    kind := "model_completed"
    toolCalls := #[call "call-1" "read"]
  }
  assert (awaitingTool.phase.label == "waiting_tool") "safe tool must skip approval"
  assert (effects.size == 1 && effects[0]!.kind == "invoke_tool") "safe tool must invoke once"

private def testUnsafeToolNeedsApproval : IO Unit := do
  let (submitted, _) ← step {} { kind := "submit", text? := some "Update the file" }
  let (awaitingApproval, effects) ← step submitted {
    kind := "model_completed"
    toolCalls := #[call "call-2" "write"]
  }
  assert (awaitingApproval.phase.label == "waiting_approval") "write must need approval"
  assert (effects.size == 1 && effects[0]!.kind == "request_approval") "write must request approval"
  let (awaitingTool, approvalEffects) ← step awaitingApproval {
    kind := "approval_result"
    callId? := some "call-2"
    approved? := some true
  }
  assert (awaitingTool.phase.label == "waiting_tool") "approved write must enter tool phase"
  assert (approvalEffects.size == 1 && approvalEffects[0]!.kind == "invoke_tool") "approved write must invoke"

private def testDeniedToolReturnsToModel : IO Unit := do
  let (submitted, _) ← step {} { kind := "submit", text? := some "Delete it" }
  let (awaitingApproval, _) ← step submitted {
    kind := "model_completed"
    toolCalls := #[call "call-3" "bash"]
  }
  let (next, effects) ← step awaitingApproval {
    kind := "approval_result"
    callId? := some "call-3"
    approved? := some false
  }
  assert (next.phase.label == "waiting_model") "denied tool must continue the agent turn"
  assert (effects.size == 1 && effects[0]!.kind == "request_model") "denied tool must request a follow-up model response"
  assert (next.messages.size == 3 && next.messages[2]!.isError) "denial must be retained as a tool result"

private def testAbortClosesEveryPendingToolCall : IO Unit := do
  let (submitted, _) ← step {} { kind := "submit", text? := some "Run two tools" }
  let (awaitingApproval, _) ← step submitted {
    kind := "model_completed"
    toolCalls := #[call "call-4" "write", call "call-5" "bash"]
  }
  let (aborted, effects) ← step awaitingApproval { kind := "abort" }
  assert (aborted.phase.label == "idle") "abort must restore idle phase"
  assert effects.isEmpty "abort must emit no external effect"
  assert (aborted.messages.size == 4) "abort must append one result for each pending tool call"
  assert (aborted.messages[2]!.toolCallId? == some "call-4") "first cancelled tool result must match"
  assert (aborted.messages[3]!.toolCallId? == some "call-5") "second cancelled tool result must match"
  let (resubmitted, nextEffects) ← step aborted { kind := "submit", text? := some "Continue" }
  assert (resubmitted.phase.label == "waiting_model") "session must accept a prompt after abort"
  assert (nextEffects.size == 1 && nextEffects[0]!.kind == "request_model") "resubmit must request the model"

private def testReadOnlyGitUsesVerifiedEffect : IO Unit := do
  let (configured, _) ← step {} { kind := "configure_tools", safeTools := #["read", "git_read"] }
  let (submitted, _) ← step configured { kind := "submit", text? := some "Inspect the repository" }
  let (awaitingTool, effects) ← step submitted {
    kind := "model_completed"
    toolCalls := #[bashCall "call-git-read" "git status --short"]
  }
  assert (awaitingTool.phase.label == "waiting_tool") "read-only Git must skip approval"
  assert (awaitingTool.pendingCalls[0]!.name == "git_read") "Git status must lower to git_read"
  assert (effects[0]!.call?.map (·.name) == some "git_read") "executor must receive the lowered Git call"

private def testMutatingGitRequiresApproval : IO Unit := do
  let (configured, _) ← step {} { kind := "configure_tools", safeTools := #["read", "git_read"] }
  let (submitted, _) ← step configured { kind := "submit", text? := some "Stage Main.lean" }
  let (awaitingApproval, effects) ← step submitted {
    kind := "model_completed"
    toolCalls := #[bashCall "call-git-write" "git add -- Main.lean"]
  }
  assert (awaitingApproval.phase.label == "waiting_approval") "mutating Git must require approval"
  assert (awaitingApproval.pendingCalls[0]!.name == "git_write") "Git add must lower to git_write"
  assert (effects[0]!.call?.map (·.name) == some "git_write") "approval must carry the lowered Git call"

private def testCompoundGitFallsBackToBash : IO Unit := do
  let (configured, _) ← step {} { kind := "configure_tools", safeTools := #["read", "git_read"] }
  let (submitted, _) ← step configured { kind := "submit", text? := some "Inspect and reset" }
  let (awaitingApproval, _) ← step submitted {
    kind := "model_completed"
    toolCalls := #[bashCall "call-git-compound" "git status && git reset --hard"]
  }
  assert (awaitingApproval.pendingCalls[0]!.name == "bash") "compound shell input must not be misclassified"

private def testGitStaysInBashWithoutCapability : IO Unit := do
  let (configured, _) ← step {} { kind := "configure_tools", safeTools := #["read"] }
  let (submitted, _) ← step configured { kind := "submit", text? := some "Inspect the repository" }
  let (awaitingApproval, _) ← step submitted {
    kind := "model_completed"
    toolCalls := #[bashCall "call-no-git" "git status --short"]
  }
  assert (awaitingApproval.pendingCalls[0]!.name == "bash")
    "Git must stay in bash when no Git plugin capability was configured"

private def testQuotedCommitMessageParses : IO Unit := do
  match parseGitCommand "git commit -m \"verified commit\"" with
  | some (.commit message) => assert (message == "verified commit") "quoted commit message must be preserved"
  | _ => throw (IO.userError "quoted Git commit must parse")


private def testAutoPermissionAllowsSimpleReads : IO Unit := do
  let (configured, _) ← step {} {
    kind := "configure_tools"
    safeTools := #["read", "git_read"]
    permissionMode := "auto"
  }
  let (submitted, _) ← step configured { kind := "submit", text? := some "Inspect files" }
  let (pwdState, pwdEffects) ← step submitted {
    kind := "model_completed"
    toolCalls := #[bashCall "call-auto-pwd" "pwd"]
  }
  assert (pwdState.phase.label == "waiting_tool") "auto mode must allow pwd"
  assert (pwdEffects[0]!.kind == "invoke_tool") "auto pwd must invoke directly"
  let (submittedAgain, _) ← step configured { kind := "submit", text? := some "List files" }
  let (lsState, _) ← step submittedAgain {
    kind := "model_completed"
    toolCalls := #[bashCall "call-auto-ls" "ls -la ."]
  }
  assert (lsState.phase.label == "waiting_tool") "auto mode must allow ls in cwd"

private def testAutoPermissionRejectsUnsafeReads : IO Unit := do
  let (configured, _) ← step {} {
    kind := "configure_tools"
    safeTools := #["read", "git_read"]
    permissionMode := "auto"
  }
  for command in ["cat src/Main.lean", "cat /etc/passwd", "cat ../secret", "cat file; rm file", "ls -L ."] do
    let (submitted, _) ← step configured { kind := "submit", text? := some "Unsafe read" }
    let (next, effects) ← step submitted {
      kind := "model_completed"
      toolCalls := #[bashCall "call-auto-reject" command]
    }
    assert (next.phase.label == "waiting_approval") s!"auto mode must prompt for {command}"
    assert (effects[0]!.kind == "request_approval") "unsafe auto command must request approval"

private def testAutoPermissionRejectsLsOperandsAndGlobs : IO Unit := do
  let (configured, _) ← step {} {
    kind := "configure_tools"
    safeTools := #["read", "git_read"]
    permissionMode := "auto"
  }
  for command in ["ls -- -private", "ls -*", "ls *", "ls -private"] do
    let (submitted, _) ← step configured { kind := "submit", text? := some "Unsafe ls" }
    let (next, effects) ← step submitted {
      kind := "model_completed"
      toolCalls := #[bashCall "call-auto-ls-reject" command]
    }
    assert (next.phase.label == "waiting_approval") s!"auto mode must prompt for {command}"
    assert (effects[0]!.kind == "request_approval") "unsafe ls must request approval"

private def testPermissionModesFailClosedAndYolo : IO Unit := do
  let (askState, _) ← step {} {
    kind := "configure_tools"
    safeTools := #["read"]
    permissionMode := "unknown"
  }
  let (askSubmitted, _) ← step askState { kind := "submit", text? := some "Run pwd" }
  let (askNext, _) ← step askSubmitted {
    kind := "model_completed"
    toolCalls := #[bashCall "call-ask" "pwd"]
  }
  assert (askNext.phase.label == "waiting_approval") "unknown permission mode must fail closed"

  let (yoloState, _) ← step {} {
    kind := "configure_tools"
    safeTools := #["read"]
    permissionMode := "yolo"
  }
  let (yoloSubmitted, _) ← step yoloState { kind := "submit", text? := some "Write" }
  let (yoloNext, yoloEffects) ← step yoloSubmitted {
    kind := "model_completed"
    toolCalls := #[call "call-yolo" "write"]
  }
  assert (yoloNext.phase.label == "waiting_tool") "yolo mode must allow write"
  assert (yoloEffects[0]!.kind == "invoke_tool") "yolo write must invoke directly"

private def testLegacySessionDefaultsToAsk : IO Unit := do
  let legacy := Json.mkObj [
    ("phase", toJson Phase.idle),
    ("messages", toJson (#[] : Array ChatMessage)),
    ("pendingCalls", toJson (#[] : Array ToolCall)),
    ("currentCall", toJson (0 : Nat)),
    ("safeTools", toJson (#[] : Array String))
  ]
  match State.fromJsonWithDefaults legacy with
  | .ok state => assert (state.permissionMode == "ask") "legacy session must default to ask"
  | .error error => throw (IO.userError s!"legacy session failed to decode: {error}")
def main : IO Unit := do
  testSafeToolFlowsDirectlyToExecutor
  testUnsafeToolNeedsApproval
  testDeniedToolReturnsToModel
  testAbortClosesEveryPendingToolCall
  testReadOnlyGitUsesVerifiedEffect
  testMutatingGitRequiresApproval
  testCompoundGitFallsBackToBash
  testGitStaysInBashWithoutCapability
  testQuotedCommitMessageParses
  testAutoPermissionAllowsSimpleReads
  testAutoPermissionRejectsUnsafeReads
  testAutoPermissionRejectsLsOperandsAndGlobs
  testPermissionModesFailClosedAndYolo
  testLegacySessionDefaultsToAsk
  IO.println "MyCode core tests passed"
