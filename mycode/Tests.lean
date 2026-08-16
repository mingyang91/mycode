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

private def callWithArgs (callId name : String) (arguments : Json) : ToolCall := {
  callId
  name
  arguments
}

private def bashCall (callId command : String) : ToolCall := {
  callId
  name := "bash"
  arguments := Json.mkObj [("command", toJson command)]
}

private def completeTurn (state : State) (prompt answer : String) (inputTokens : Nat) : IO State := do
  let (submitted, _) ← step state { kind := "submit", text? := some prompt }
  let (completed, _) ← step submitted {
    kind := "model_completed"
    content? := some answer
    inputTokens
  }
  pure completed

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

private def testSteerRestartsWaitingModel : IO Unit := do
  let (submitted, _) ← step {} { kind := "submit", text? := some "Use the old plan" }
  let (steered, effects) ← step submitted { kind := "steer", text? := some "Use the new plan" }
  assert (steered.phase.label == "waiting_model") "steer must keep the model phase active"
  assert (effects.size == 1 && effects[0]!.kind == "request_model")
    "steer during model work must request a replacement response"
  assert (steered.messages.size == 2) "model steer must retain both user instructions"
  assert (steered.messages[1]!.content == "Use the new plan") "steer text must enter the main transcript"
  assert steered.pendingSteers.isEmpty "model steer must be consumed before the replacement request"

private def testSteerFinishesCurrentToolAndSkipsTheRest : IO Unit := do
  let (configured, _) ← step {} { kind := "configure_tools", safeTools := #["read"] }
  let (submitted, _) ← step configured { kind := "submit", text? := some "Read both files" }
  let (running, _) ← step submitted {
    kind := "model_completed"
    toolCalls := #[call "call-steer-current" "read", call "call-steer-skipped" "read"]
  }
  let (queued, steerEffects) ← step running { kind := "steer", text? := some "Read only the first file" }
  assert steerEffects.isEmpty "steer must not interrupt a tool with an unknown outcome"
  assert (queued.pendingSteers == #["Read only the first file"]) "tool steer must persist until the result arrives"
  let (next, effects) ← step queued {
    kind := "tool_completed"
    callId? := some "call-steer-current"
    content? := some "first contents"
  }
  assert (next.phase.label == "waiting_model") "tool steer must return control to the model"
  assert (effects.size == 1 && effects[0]!.kind == "request_model") "tool steer must request the model once"
  assert (next.pendingCalls.isEmpty && next.pendingSteers.isEmpty) "consumed steer state must be cleared"
  assert (next.messages.size == 5) "tool steer must retain the result, skip, and instruction"
  assert (next.messages[2]!.toolCallId? == some "call-steer-current" && !next.messages[2]!.isError)
    "the running tool result must be retained"
  assert (next.messages[3]!.toolCallId? == some "call-steer-skipped" && next.messages[3]!.isError)
    "the remaining tool must receive a synthetic result"
  assert (next.messages[4]!.role == "user" && next.messages[4]!.content == "Read only the first file")
    "steer text must follow every tool result"

private def testSteerReplacesPendingApproval : IO Unit := do
  let (submitted, _) ← step {} { kind := "submit", text? := some "Change two files" }
  let (approval, _) ← step submitted {
    kind := "model_completed"
    toolCalls := #[call "call-steer-denied-1" "write", call "call-steer-denied-2" "bash"]
  }
  let (steered, effects) ← step approval { kind := "steer", text? := some "Do not change files" }
  assert (steered.phase.label == "waiting_model") "approval steer must resume the model"
  assert (effects.size == 1 && effects[0]!.kind == "request_model") "approval steer must request the model"
  assert (steered.messages.size == 5) "approval steer must close every proposed tool"
  assert (steered.messages[2]!.isError && steered.messages[3]!.isError)
    "approval steer must synthesize every missing tool result"
  assert (steered.messages[4]!.role == "user" && steered.messages[4]!.content == "Do not change files")
    "approval steer must append the new instruction after tool results"

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

private def testTodoBuiltinPersistsLifecycle : IO Unit := do
  let phases : Array TodoPhase := #[{
    name := "Implementation"
    tasks := #[
      { content := "Add protocol" },
      { content := "Run verification" }
    ]
  }]
  let (submitted, _) ← step {} { kind := "submit", text? := some "Track this work" }
  let (initialized, initEffects) ← step submitted {
    kind := "model_completed"
    toolCalls := #[callWithArgs "call-todo-init" "todo" <| Json.mkObj [
      ("op", toJson "init"),
      ("phases", toJson phases)
    ]]
  }
  assert (initialized.phase.label == "waiting_model") "todo init must continue the model turn"
  assert (initEffects.size == 1 && initEffects[0]!.kind == "request_model")
    "todo init must request a follow-up model response"
  assert (initialized.todos == phases) "todo init must persist structured phases"
  assert (initialized.messages.back!.role == "tool" && !initialized.messages.back!.isError)
    "todo init must close its tool call successfully"

  let (started, _) ← step initialized {
    kind := "model_completed"
    toolCalls := #[callWithArgs "call-todo-start" "todo" <| Json.mkObj [
      ("op", toJson "start"),
      ("task", toJson "Add protocol")
    ]]
  }
  assert (started.todos[0]!.tasks[0]!.status == "in_progress") "todo start must select the active task"

  let (completed, _) ← step started {
    kind := "model_completed"
    toolCalls := #[callWithArgs "call-todo-done" "todo" <| Json.mkObj [
      ("op", toJson "done"),
      ("task", toJson "Add protocol")
    ]]
  }
  assert (completed.todos[0]!.tasks[0]!.status == "completed") "todo done must persist completion"

private def testHumanTodoReplacementIsValidated : IO Unit := do
  let phases : Array TodoPhase := #[{
    name := "Review"
    tasks := #[{ content := "Inspect the plan", status := "in_progress" }]
  }]
  let (updated, effects) ← step {} { kind := "replace_todos", todos := phases }
  assert (updated.todos == phases) "human todo replacement must become canonical state"
  assert effects.isEmpty "human todo replacement must not emit an external effect"

private def testPlanReviewBlocksWritesAndSupportsRevision : IO Unit := do
  let (configured, _) ← step {} {
    kind := "configure_tools"
    safeTools := #["read"]
    permissionMode := "yolo"
  }
  let (planning, effects) ← step configured { kind := "enter_plan", text? := some "Design safer storage" }
  assert (planning.plan.enabled && planning.plan.status == "draft") "enter_plan must enable draft mode"
  assert (effects.size == 1 && effects[0]!.kind == "request_model") "enter_plan must request the model"

  let firstPlan := "# Safer storage\n\n1. Inspect the current format.\n2. Propose a migration."
  let (review, reviewEffects) ← step planning {
    kind := "model_completed"
    toolCalls := #[
      call "call-plan-write" "write",
      callWithArgs "call-plan-propose" "plan" <| Json.mkObj [
        ("op", toJson "propose"),
        ("content", toJson firstPlan)
      ]
    ]
  }
  assert (review.phase.label == "waiting_plan_review") "plan proposal must wait for human review"
  assert (reviewEffects.size == 1 && reviewEffects[0]!.kind == "request_plan_review")
    "plan proposal must emit only the review effect"
  assert (review.messages[2]!.isError && review.messages[2]!.toolCallId? == some "call-plan-write")
    "plan mode must block workspace writes even in yolo"
  assert (review.plan.revision == 1 && review.plan.content == firstPlan) "proposal must persist revision one"

  let editedPlan := firstPlan ++ "\n3. Verify rollback."
  let (edited, editEffects) ← step review {
    kind := "plan_review_result"
    content? := some editedPlan
  }
  assert (edited.phase.label == "waiting_plan_review" && edited.plan.revision == 2)
    "human plan edits must create a new review revision"
  assert (editEffects.size == 1 && editEffects[0]!.kind == "request_plan_review")
    "edited plan must return to review"

  let (refining, refineEffects) ← step edited {
    kind := "plan_review_result"
    text? := some "Use a bounded migration batch."
  }
  assert (refining.phase.label == "waiting_model" && refining.plan.status == "draft")
    "review feedback must return to planning"
  assert (refineEffects.size == 1 && refineEffects[0]!.kind == "request_model")
    "review feedback must request a replacement plan"

  let finalPlan := editedPlan ++ "\n4. Bound every batch."
  let (finalReview, _) ← step refining {
    kind := "model_completed"
    toolCalls := #[callWithArgs "call-plan-final" "plan" <| Json.mkObj [
      ("op", toJson "propose"),
      ("content", toJson finalPlan)
    ]]
  }
  let (approved, approvalEffects) ← step finalReview {
    kind := "plan_review_result"
    approved? := some true
  }
  assert (!approved.plan.enabled && approved.plan.status == "approved")
    "plan approval must leave plan mode"
  assert (approved.phase.label == "waiting_model") "approved plan must begin execution"
  assert (approvalEffects.size == 1 && approvalEffects[0]!.kind == "request_model")
    "approved plan must request the execution turn"

private def testManualCompactionPreservesHistory : IO Unit := do
  let first ← completeTurn {} "First request" "First answer" 100
  let second ← completeTurn first "Second request" "Second answer" 200
  let third ← completeTurn second "Third request" "Third answer" 300
  assert (third.messages.size == 6 && third.compaction.lastInputTokens == 300)
    "model completion must retain full history and input usage"

  let (compacting, effects) ← step third {
    kind := "start_compaction"
    text? := some "Preserve storage decisions."
    inputTokens := 300
  }
  assert (compacting.phase.label == "waiting_compaction") "manual compact must enter compaction phase"
  assert (effects.size == 1 && effects[0]!.kind == "request_compaction")
    "manual compact must request one summary"
  let pending := compacting.compaction.pending?.get!
  assert (pending.firstKeptMessage == 2 && !pending.automatic && !pending.continueAfter)
    "manual compact must keep the latest two complete user turns"

  let summary := "The first request established the storage constraint."
  let (compacted, completedEffects) ← step compacting {
    kind := "compaction_completed"
    content? := some summary
  }
  assert (compacted.phase.label == "idle" && completedEffects.isEmpty)
    "manual compact must return to idle"
  assert (compacted.messages.size == 6) "compaction must never delete canonical transcript messages"
  assert (compacted.compaction.revision == 1 && compacted.compaction.summary == summary)
    "compaction must persist its summary and revision"
  assert (compacted.compaction.firstKeptMessage == 2 && compacted.compaction.lastInputTokens == 0)
    "compaction must persist its cut and reset the threshold observation"

private def testAutomaticCompactionResumesModelAndRecoversFailure : IO Unit := do
  let first ← completeTurn {} "First request" "First answer" 100
  let second ← completeTurn first "Second request" "Second answer" 200
  let third ← completeTurn second "Third request" "Third answer" 900
  let (waiting, _) ← step third { kind := "submit", text? := some "Fourth request" }

  let (compacting, effects) ← step waiting {
    kind := "start_compaction"
    inputTokens := 900
    automatic := true
  }
  assert (effects.size == 1 && effects[0]!.kind == "request_compaction")
    "automatic compact must request a summary"
  let pending := compacting.compaction.pending?.get!
  assert (pending.automatic && pending.continueAfter) "pre-request compact must remember to resume"

  let (recovered, recoveryEffects) ← step compacting { kind := "compaction_failed" }
  assert (recovered.phase.label == "waiting_model") "failed pre-request compact must restore model phase"
  assert (recoveryEffects.size == 1 && recoveryEffects[0]!.kind == "request_model")
    "failed pre-request compact must continue without looping"
  assert (recovered.compaction.pending?.isNone && recovered.compaction.lastInputTokens == 0)
    "failed compact must clear pending state and its trigger observation"

private def testAbortClearsPendingCompaction : IO Unit := do
  let first ← completeTurn {} "First request" "First answer" 100
  let second ← completeTurn first "Second request" "Second answer" 200
  let third ← completeTurn second "Third request" "Third answer" 300
  let (compacting, _) ← step third { kind := "start_compaction" }
  let (aborted, effects) ← step compacting { kind := "abort" }
  assert (aborted.phase.label == "idle" && effects.isEmpty) "abort must stop manual compaction"
  assert aborted.compaction.pending?.isNone "abort must clear pending compaction state"

private def testLegacySessionDefaultsToAsk : IO Unit := do
  let legacy := Json.mkObj [
    ("phase", toJson Phase.idle),
    ("messages", toJson (#[] : Array ChatMessage)),
    ("pendingCalls", toJson (#[] : Array ToolCall)),
    ("currentCall", toJson (0 : Nat)),
    ("safeTools", toJson (#[] : Array String))
  ]
  match State.fromJsonWithDefaults legacy with
  | .ok state =>
    assert (state.permissionMode == "ask") "legacy session must default to ask"
    assert state.pendingSteers.isEmpty "legacy session must default to no queued steer"
    assert (!state.plan.enabled && state.plan.revision == 0) "legacy session must default to no plan"
    assert state.todos.isEmpty "legacy session must default to no todos"
    assert (state.compaction.revision == 0 && state.compaction.pending?.isNone)
      "legacy session must default to no compaction"
  | .error error => throw (IO.userError s!"legacy session failed to decode: {error}")
def main : IO Unit := do
  testSafeToolFlowsDirectlyToExecutor
  testUnsafeToolNeedsApproval
  testDeniedToolReturnsToModel
  testAbortClosesEveryPendingToolCall
  testSteerRestartsWaitingModel
  testSteerFinishesCurrentToolAndSkipsTheRest
  testSteerReplacesPendingApproval
  testReadOnlyGitUsesVerifiedEffect
  testMutatingGitRequiresApproval
  testCompoundGitFallsBackToBash
  testGitStaysInBashWithoutCapability
  testQuotedCommitMessageParses
  testAutoPermissionAllowsSimpleReads
  testAutoPermissionRejectsUnsafeReads
  testAutoPermissionRejectsLsOperandsAndGlobs
  testPermissionModesFailClosedAndYolo
  testTodoBuiltinPersistsLifecycle
  testHumanTodoReplacementIsValidated
  testPlanReviewBlocksWritesAndSupportsRevision
  testManualCompactionPreservesHistory
  testAutomaticCompactionResumesModelAndRecoversFailure
  testAbortClearsPendingCompaction
  testLegacySessionDefaultsToAsk
  IO.println "MyCode core tests passed"
