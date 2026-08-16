import MyCode.Git
import MyCode.Planning

namespace MyCode

private def currentTool? (state : State) : Option ToolCall :=
  state.pendingCalls[state.currentCall]?

private def normalizedPermissionMode (mode : String) : String :=
  if ["ask", "auto", "yolo"].contains mode then mode else "ask"

private def isSafeTool (state : State) (call : ToolCall) : Bool :=
  state.permissionMode == "yolo" ||
    state.safeTools.any (fun name => name == call.name) ||
    (state.permissionMode == "auto" && call.autoPermissionAllowed)

private def appendMessage (state : State) (message : ChatMessage) : State :=
  { state with messages := state.messages.push message }


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

private def advanceWithFuel (state : State) (fuel : Nat) : State × Array Effect :=
  match currentTool? state with
  | none =>
    ({ state with phase := .waitingModel, pendingCalls := #[], currentCall := 0 }, #[Effect.requestModel])
  | some call =>
    match applyBuiltinTool? state call with
    | some result =>
      let completed := { finishTool result.state call result.content result.isError with
        currentCall := state.currentCall + 1 }
      if result.requestPlanReview && !result.isError then
        let unanswered := completed.pendingCalls.toList.drop completed.currentCall
        let closed := unanswered.foldl (fun next pending =>
          finishTool next pending "Tool call skipped because the plan is awaiting review." true) completed
        ({ closed with phase := .waitingPlanReview, pendingCalls := #[], currentCall := 0 },
          #[Effect.requestPlanReview])
      else
        match fuel with
        | 0 => ({ completed with phase := .idle, pendingCalls := #[], currentCall := 0 }, #[])
        | remaining + 1 => advanceWithFuel completed remaining
    | none =>
      if state.plan.enabled then
        if state.safeTools.any (fun name => name == call.name) || call.autoPermissionAllowed then
          ({ state with phase := .waitingTool call.callId }, #[Effect.invokeTool call])
        else
          let denied := { finishTool state call "Tool is unavailable while plan mode is active." true with
            currentCall := state.currentCall + 1 }
          match fuel with
          | 0 => ({ denied with phase := .idle, pendingCalls := #[], currentCall := 0 }, #[])
          | remaining + 1 => advanceWithFuel denied remaining
      else if isSafeTool state call then
        ({ state with phase := .waitingTool call.callId }, #[Effect.invokeTool call])
      else
        ({ state with phase := .waitingApproval call.callId }, #[Effect.requestApproval call])

private def advance (state : State) : State × Array Effect :=
  advanceWithFuel state (state.pendingCalls.size + 1)

private def finishPendingTools (state : State) (content : String) : State :=
  let unanswered := state.pendingCalls.toList.drop state.currentCall
  unanswered.foldl (fun next call => finishTool next call content true) state

private def resumeWithPendingSteers (state : State) : State :=
  let completed := finishPendingTools state "Tool call skipped because the user steered the active turn."
  let steered := state.pendingSteers.foldl (fun next text =>
    appendMessage next { role := "user", content := text }) completed
  { steered with
    phase := .waitingModel
    pendingCalls := #[]
    currentCall := 0
    pendingSteers := #[]
  }

private def abortPendingTools (state : State) : State :=
  let completed := finishPendingTools state "Tool execution was cancelled before a result was observed."
  { completed with
    phase := .idle
    pendingCalls := #[]
    currentCall := 0
    pendingSteers := #[]
    compaction := { state.compaction with pending? := none }
  }

private def requireIdle (state : State) (action : String) : Except String Unit :=
  match state.phase with
  | .idle => pure ()
  | _ => throw s!"cannot {action} while the agent is {state.phase.label}"

private def requireCallId (expected : String) (received? : Option String) : Except String Unit :=
  match received? with
  | some received =>
    if received == expected then pure () else throw "event call id does not match the active call"
  | none => throw "event is missing callId"

private def compactionCut? (state : State) : Option Nat :=
  let userIndices := (List.range state.messages.size).filter fun index =>
    match state.messages[index]? with
    | some message => message.role == "user"
    | none => false
  match userIndices.reverse with
  | _latest :: secondLatest :: _ =>
    if secondLatest > state.compaction.firstKeptMessage then some secondLatest else none
  | _ => none

private def compactionInputTokens (state : State) (event : Event) : Nat :=
  if event.inputTokens > 0 then event.inputTokens else state.compaction.lastInputTokens

private def validateCompactionInstructions (instructions? : Option String) : Except String Unit := do
  match instructions? with
  | some instructions =>
    if instructions.length > 1000 then throw "compaction instructions exceed 1000 characters"
  | none => pure ()

private def validateCompactionSummary (summary : String) : Except String Unit := do
  if summary.trimAscii.isEmpty then throw "compaction summary must not be empty"
  if summary.length > 65536 then throw "compaction summary exceeds 65536 characters"

/-- The only business decider. It mutates no external system: Rust realizes each emitted effect
and supplies another event with the observed result. -/
public def transition (state : State) (event : Event) : Except String (State × Array Effect) := do
  match event.kind with
  | "configure_tools" =>
    match state.phase with
    | .idle => pure ({ state with
        safeTools := event.safeTools
        permissionMode := normalizedPermissionMode event.permissionMode
      }, #[])
    | _ => throw "tool catalogue cannot change while the agent is running"
  | "enter_plan" =>
    requireIdle state "enter plan mode"
    match event.text? with
    | some text =>
      if text.trimAscii.isEmpty then throw "plan request must not be empty"
      else
        let planning := { state with
          phase := .waitingModel
          plan := { state.plan with enabled := true, status := "draft" }
        }
        let next := appendMessage planning { role := "user", content := text }
        pure (next, #[Effect.requestModel])
    | none => throw "enter_plan event is missing text"
  | "replace_todos" =>
    match state.phase with
    | .idle | .waitingPlanReview =>
      validateTodoPhases event.todos
      pure ({ state with todos := event.todos }, #[])
    | _ => throw "cannot replace todos while the agent is running"
  | "start_compaction" =>
    let continueAfter ← match state.phase with
      | .idle => pure false
      | .waitingModel => pure true
      | _ => throw s!"cannot compact while the agent is {state.phase.label}"
    validateCompactionInstructions event.text?
    match compactionCut? state with
    | none => throw "not enough uncompacted history to compact"
    | some firstKeptMessage =>
      let pending : PendingCompaction := {
        firstKeptMessage
        tokensBefore := compactionInputTokens state event
        instructions? := event.text?
        automatic := event.automatic
        continueAfter
      }
      pure ({ state with
        phase := .waitingCompaction
        compaction := { state.compaction with pending? := some pending }
      }, #[Effect.requestCompaction])
  | "compaction_completed" =>
    match state.phase, state.compaction.pending?, event.content? with
    | .waitingCompaction, some pending, some summary =>
      validateCompactionSummary summary
      let compaction : CompactionState := {
        revision := state.compaction.revision + 1
        summary
        firstKeptMessage := pending.firstKeptMessage
        tokensBefore := pending.tokensBefore
        lastInputTokens := 0
        pending? := none
      }
      if pending.continueAfter then
        pure ({ state with phase := .waitingModel, compaction }, #[Effect.requestModel])
      else
        pure ({ state with phase := .idle, compaction }, #[])
    | .waitingCompaction, some _, none => throw "compaction_completed event is missing content"
    | .waitingCompaction, none, _ => throw "compaction completed without pending state"
    | _, _, _ => throw "compaction_completed arrived without active compaction"
  | "compaction_failed" =>
    match state.phase, state.compaction.pending? with
    | .waitingCompaction, some pending =>
      let compaction := { state.compaction with lastInputTokens := 0, pending? := none }
      if pending.continueAfter then
        pure ({ state with phase := .waitingModel, compaction }, #[Effect.requestModel])
      else
        pure ({ state with phase := .idle, compaction }, #[])
    | .waitingCompaction, none => throw "compaction failed without pending state"
    | _, _ => throw "compaction_failed arrived without active compaction"
  | "submit" =>
    requireIdle state "submit a prompt"
    match event.text? with
    | some text =>
      if text.isEmpty then throw "prompt must not be empty"
      else
        let next := appendMessage { state with phase := .waitingModel } { role := "user", content := text }
        pure (next, #[Effect.requestModel])
    | none => throw "submit event is missing text"
  | "steer" =>
    match event.text? with
    | some text =>
      if text.isEmpty then throw "steer event text must not be empty"
      else
        let queued := { state with pendingSteers := state.pendingSteers.push text }
        match state.phase with
        | .idle => throw "cannot steer while the agent is idle"
        | .waitingTool _ => pure (queued, #[])
        | .waitingModel => pure (resumeWithPendingSteers queued, #[Effect.requestModel])
        | .waitingApproval _ => pure (resumeWithPendingSteers queued, #[Effect.requestModel])
        | .waitingPlanReview => throw "cannot steer while a plan is awaiting review"
        | .waitingCompaction => throw "cannot steer while compaction is running"
    | none => throw "steer event is missing text"
  | "model_completed" =>
    match state.phase with
    | .waitingModel =>
      let content := event.content?.getD ""
      let withAssistant := appendMessage { state with
        compaction := { state.compaction with lastInputTokens := event.inputTokens }
      } {
        role := "assistant"
        content
        toolCalls := event.toolCalls
      }
      if event.toolCalls.isEmpty then
        pure ({ withAssistant with phase := .idle }, #[])
      else
        let pendingCalls :=
          if state.safeTools.any (· == "git_read") then lowerGitToolCalls event.toolCalls
          else event.toolCalls
        pure <| advance { withAssistant with pendingCalls, currentCall := 0 }
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
        let next := { finishTool state call content (event.isError?.getD false) with
          currentCall := state.currentCall + 1 }
        if next.pendingSteers.isEmpty then pure <| advance next
        else pure (resumeWithPendingSteers next, #[Effect.requestModel])
      | none => throw "tool_completed arrived without a pending tool call"
    | _ => throw "tool_completed arrived without a running tool"
  | "plan_review_result" =>
    match state.phase with
    | .waitingPlanReview =>
      match event.approved?, event.content?, event.text? with
      | some true, _, _ =>
        let plan := { state.plan with enabled := false, status := "approved" }
        let approved := appendMessage { state with phase := .waitingModel, plan } {
          role := "user"
          content := s!"Plan revision {plan.revision} approved. Execute it and keep the todo list current."
        }
        pure (approved, #[Effect.requestModel])
      | _, some content, _ =>
        validatePlanContent content
        let plan := {
          enabled := true
          revision := state.plan.revision + 1
          status := "review"
          content
        }
        pure ({ state with plan, phase := .waitingPlanReview }, #[Effect.requestPlanReview])
      | _, _, some feedback =>
        if feedback.trimAscii.isEmpty then throw "plan feedback must not be empty"
        else
          let plan := { state.plan with enabled := true, status := "draft" }
          let revised := appendMessage { state with phase := .waitingModel, plan } {
            role := "user"
            content := s!"Plan review feedback: {feedback}"
          }
          pure (revised, #[Effect.requestModel])
      | some false, none, none =>
        pure ({ state with phase := .idle, plan := { state.plan with status := "draft" } }, #[])
      | none, none, none => throw "plan_review_result is missing a decision"
    | _ => throw "plan_review_result arrived without a pending plan review"
  | "abort" =>
    pure (abortPendingTools state, #[])
  | _ => throw s!"unknown event kind: {event.kind}"


end MyCode
