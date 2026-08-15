import LeanAgentCore.Protocol

namespace LeanAgentCore

private def currentTool? (state : State) : Option ToolCall :=
  state.pendingCalls[state.currentCall]?

private def isSafeTool (state : State) (call : ToolCall) : Bool :=
  state.safeTools.any (fun name => name == call.name)

private def appendMessage (state : State) (message : ChatMessage) : State :=
  { state with messages := state.messages.push message }

private def advance (state : State) : State × Array Effect :=
  match currentTool? state with
  | none =>
    ({ state with phase := .waitingModel, pendingCalls := #[], currentCall := 0 }, #[Effect.requestModel])
  | some call =>
    if isSafeTool state call then
      ({ state with phase := .waitingTool call.callId }, #[Effect.invokeTool call])
    else
      ({ state with phase := .waitingApproval call.callId }, #[Effect.requestApproval call])

private def denyCurrentTool (state : State) (call : ToolCall) : State :=
  appendMessage state {
    role := "tool"
    content := "Tool call denied by user."
    toolCallId? := some call.callId
    isError := true
  }

private def finishTool (state : State) (call : ToolCall) (content : String) (isError : Bool) : State :=
  appendMessage state {
    role := "tool"
    content
    toolCallId? := some call.callId
    isError
  }

private def abortPendingTools (state : State) : State :=
  let unanswered := state.pendingCalls.toList.drop state.currentCall
  let completed := unanswered.foldl (fun next call =>
    finishTool next call "Tool execution was cancelled before a result was observed." true) state
  { completed with phase := .idle, pendingCalls := #[], currentCall := 0 }

private def requireIdle (state : State) (action : String) : Except String Unit :=
  match state.phase with
  | .idle => pure ()
  | _ => throw s!"cannot {action} while the agent is {state.phase.label}"

private def requireCallId (expected : String) (received? : Option String) : Except String Unit :=
  match received? with
  | some received =>
    if received == expected then pure () else throw "event call id does not match the active call"
  | none => throw "event is missing callId"

/-- The only business decider. It mutates no external system: Rust realizes each emitted effect
and supplies another event with the observed result. -/
public def transition (state : State) (event : Event) : Except String (State × Array Effect) := do
  match event.kind with
  | "configure_tools" =>
    match state.phase with
    | .idle => pure ({ state with safeTools := event.safeTools }, #[])
    | _ => throw "tool catalogue cannot change while the agent is running"
  | "submit" =>
    requireIdle state "submit a prompt"
    match event.text? with
    | some text =>
      if text.isEmpty then throw "prompt must not be empty"
      else
        let next := appendMessage { state with phase := .waitingModel } { role := "user", content := text }
        pure (next, #[Effect.requestModel])
    | none => throw "submit event is missing text"
  | "model_completed" =>
    match state.phase with
    | .waitingModel =>
      let content := event.content?.getD ""
      let withAssistant := appendMessage state {
        role := "assistant"
        content
        toolCalls := event.toolCalls
      }
      if event.toolCalls.isEmpty then
        pure ({ withAssistant with phase := .idle }, #[])
      else
        pure <| advance { withAssistant with pendingCalls := event.toolCalls, currentCall := 0 }
    | _ => throw "model_completed arrived without an active model request"
  | "approval_result" =>
    match state.phase with
    | .waitingApproval expected =>
      requireCallId expected event.callId?
      match currentTool? state, event.approved? with
      | some call, some true => pure ({ state with phase := .waitingTool call.callId }, #[Effect.invokeTool call])
      | some call, some false => pure <| advance <| { denyCurrentTool state call with currentCall := state.currentCall + 1 }
      | none, _ => throw "approval was requested without a pending tool call"
      | _, none => throw "approval_result event is missing approved"
    | _ => throw "approval_result arrived without a pending approval"
  | "tool_completed" =>
    match state.phase with
    | .waitingTool expected =>
      requireCallId expected event.callId?
      match currentTool? state with
      | some call =>
        let content := event.content?.getD ""
        let next := finishTool state call content (event.isError?.getD false)
        pure <| advance { next with currentCall := state.currentCall + 1 }
      | none => throw "tool_completed arrived without a pending tool call"
    | _ => throw "tool_completed arrived without a running tool"
  | "abort" =>
    pure (abortPendingTools state, #[])
  | _ => throw s!"unknown event kind: {event.kind}"


end LeanAgentCore
