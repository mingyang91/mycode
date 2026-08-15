import Lean.Data.Json

open Lean

namespace MyCode

public def protocolVersion : Nat := 1

public structure ToolCall where
  callId : String
  name : String
  arguments : Json
  deriving Inhabited, ToJson, FromJson

public structure ChatMessage where
  role : String
  content : String
  toolCalls : Array ToolCall := #[]
  toolCallId? : Option String := none
  isError : Bool := false
  deriving Inhabited, ToJson, FromJson

public inductive Phase where
  | idle
  | waitingModel
  | waitingApproval (callId : String)
  | waitingTool (callId : String)
  deriving Inhabited, Repr, ToJson, FromJson

public def Phase.label : Phase → String
  | .idle => "idle"
  | .waitingModel => "waiting_model"
  | .waitingApproval _ => "waiting_approval"
  | .waitingTool _ => "waiting_tool"

public structure State where
  phase : Phase := .idle
  messages : Array ChatMessage := #[]
  pendingCalls : Array ToolCall := #[]
  currentCall : Nat := 0
  safeTools : Array String := #[]
  permissionMode : String := "ask"
  deriving Inhabited, ToJson, FromJson

public structure Event where
  kind : String
  text? : Option String := none
  toolCalls : Array ToolCall := #[]
  callId? : Option String := none
  approved? : Option Bool := none
  content? : Option String := none
  isError? : Option Bool := none
  safeTools : Array String := #[]
  permissionMode : String := "ask"
  deriving Inhabited, ToJson, FromJson

public structure Effect where
  kind : String
  call? : Option ToolCall := none
  deriving Inhabited, ToJson, FromJson

public structure Snapshot where
  phase : String
  messages : Array ChatMessage
  pendingCalls : Array ToolCall
  currentCall : Nat
  safeTools : Array String
  permissionMode : String
  deriving Inhabited, ToJson, FromJson

public def State.snapshot (state : State) : Snapshot := {
  phase := state.phase.label
  messages := state.messages
  pendingCalls := state.pendingCalls
  currentCall := state.currentCall
  safeTools := state.safeTools
  permissionMode := state.permissionMode
}

public def State.fromJsonWithDefaults (json : Json) : Except String State :=
  let migrated := match json with
    | .obj _ =>
      match json.getObjVal? "permissionMode" with
      | .ok _ => json
      | .error _ => json.setObjVal! "permissionMode" (toJson "ask")
    | _ => json
  fromJson? migrated

public structure Request where
  version : Nat
  requestId : String
  op : String
  event? : Option Event := none
  deriving Inhabited, ToJson, FromJson

public structure Response where
  version : Nat := protocolVersion
  requestId : String
  ok : Bool
  snapshot : Snapshot
  effects : Array Effect := #[]
  error? : Option String := none
  deriving Inhabited, ToJson, FromJson

public def Effect.requestModel : Effect := { kind := "request_model" }

public def Effect.requestApproval (call : ToolCall) : Effect := {
  kind := "request_approval"
  call? := some call
}

public def Effect.invokeTool (call : ToolCall) : Effect := {
  kind := "invoke_tool"
  call? := some call
}

end MyCode
