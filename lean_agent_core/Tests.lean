import LeanAgentCore.State

open Lean
open LeanAgentCore

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

def main : IO Unit := do
  testSafeToolFlowsDirectlyToExecutor
  testUnsafeToolNeedsApproval
  testDeniedToolReturnsToModel
  testAbortClosesEveryPendingToolCall
  IO.println "Lean agent core tests passed"
