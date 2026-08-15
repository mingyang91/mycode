import MyCode.State

open Lean
open MyCode

private def decodeRequest (line : String) : Except String Request := do
  let json ← Json.parse line
  fromJson? json

private def encodeResponse (response : Response) : String :=
  Json.compress (toJson response)

private def response (requestId : String) (state : State) (ok : Bool) (effects : Array Effect := #[])
    (error? : Option String := none) : Response := {
  requestId
  ok
  snapshot := state.snapshot
  effects
  error?
}

private def sessionPathFromArgs (args : List String) : Option System.FilePath :=
  match args with
  | "--session" :: path :: _ => some (System.FilePath.mk path)
  | _ :: rest => sessionPathFromArgs rest
  | [] => none

private def loadState (path? : Option System.FilePath) : IO State := do
  match path? with
  | none => pure {}
  | some path =>
    if !(← path.pathExists) then
      pure {}
    else
      let text ← IO.FS.readFile path
      let json ← IO.ofExcept (Json.parse text)
      IO.ofExcept (fromJson? json)

private def saveState (path? : Option System.FilePath) (state : State) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
    let temporary := path.addExtension "tmp"
    IO.FS.writeFile temporary (Json.compress (toJson state))
    IO.FS.rename temporary path

private def dispatch (state : State) (request : Request) : Except String (State × Array Effect × Bool) := do
  if request.version != protocolVersion then
    throw s!"unsupported core protocol version {request.version}"
  match request.op with
  | "event" =>
    match request.event? with
    | some event =>
      let (next, effects) ← transition state event
      pure (next, effects, false)
    | none => throw "event request is missing event"
  | "snapshot" => pure (state, #[], false)
  | "shutdown" => pure (state, #[], true)
  | _ => throw s!"unknown core request operation: {request.op}"

private partial def serve (path? : Option System.FilePath) (initial : State) : IO Unit := do
  let input ← IO.getStdin
  let output ← IO.getStdout
  let rec loop (state : State) : IO Unit := do
    let line ← input.getLine
    if line.isEmpty then
      pure ()
    else
      match decodeRequest line.trimAscii.copy with
      | .error err =>
        output.putStrLn (encodeResponse (response "invalid" state false #[] (some s!"invalid core request: {err}")))
        output.flush
        loop state
      | .ok request =>
        match dispatch state request with
        | .error err =>
          output.putStrLn (encodeResponse (response request.requestId state false #[] (some err)))
          output.flush
          loop state
        | .ok (next, effects, shutdown) =>
          try
            saveState path? next
            output.putStrLn (encodeResponse (response request.requestId next true effects))
            output.flush
            if shutdown then pure () else loop next
          catch error =>
            output.putStrLn (encodeResponse (response request.requestId state false #[] (some s!"failed to persist session: {error}")))
            output.flush
            loop state
  loop initial

def main (args : List String) : IO Unit := do
  let path? := sessionPathFromArgs args
  let state ← loadState path?
  IO.eprintln "mycode-core ready"
  serve path? state
