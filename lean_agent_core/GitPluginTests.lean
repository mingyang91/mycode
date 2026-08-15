import LeanAgentCore.Git

open Lean

private def maxFrameBytes : Nat := 1_048_576

private def assert (condition : Bool) (label : String) : IO Unit :=
  unless condition do throw (IO.userError label)

private partial def readExact (handle : IO.FS.Handle) (length : Nat) (data := ByteArray.empty) : IO ByteArray := do
  if data.size == length then
    pure data
  else
    let chunk ← handle.read (length - data.size).toUSize
    if chunk.isEmpty then throw (IO.userError "unexpected end of plugin frame")
    else readExact handle length (data ++ chunk)

private def frameLength (header : ByteArray) : Nat :=
  header[0]!.toNat * 16_777_216 +
    header[1]!.toNat * 65_536 +
    header[2]!.toNat * 256 +
    header[3]!.toNat

private def readFrame (handle : IO.FS.Handle) : IO Json := do
  let header ← readExact handle 4
  let length := frameLength header
  if length == 0 || length > maxFrameBytes then throw (IO.userError "invalid plugin frame length")
  let payload ← readExact handle length
  let some text := String.fromUTF8? payload | throw (IO.userError "plugin response was not UTF-8")
  IO.ofExcept (Json.parse text)

private def writeFrame (handle : IO.FS.Handle) (json : Json) : IO Unit := do
  let payload := json.compress.toUTF8
  if payload.isEmpty || payload.size > maxFrameBytes then throw (IO.userError "invalid plugin request frame")
  let length := payload.size
  let header := ByteArray.empty
    |>.push (length / 16_777_216).toUInt8
    |>.push (length / 65_536 % 256).toUInt8
    |>.push (length / 256 % 256).toUInt8
    |>.push (length % 256).toUInt8
  handle.write header
  handle.write payload
  handle.flush

private def request (id operation : String) (parameters : Json) : Json :=
  Json.mkObj [
    ("v", 1),
    ("id", id),
    ("op", operation),
    ("params", parameters)
  ]

private def responseOk (response : Json) : Except String Bool := do
  let value ← response.getObjVal? "ok"
  value.getBool?

private def hasTool (tools : Array Json) (name : String) : Bool :=
  tools.any fun tool =>
    match tool.getObjVal? "name" with
    | .ok (.str toolName) => toolName == name
    | _ => false

private def testGitPluginProtocol : IO Unit := do
  let plugin ← IO.FS.realPath (System.FilePath.mk ".lake/build/bin/lean_agent_git_plugin")
  let workspace ← IO.FS.realPath (System.FilePath.mk "..")
  let child ← IO.Process.spawn {
    cmd := plugin.toString
    cwd := some workspace
    stdin := .piped
    stdout := .piped
    stderr := .inherit
  }
  let (input, child) ← child.takeStdin
  writeFrame input <| request "init_1" "initialize" <| Json.mkObj [
    ("host", Json.mkObj [("name", "test"), ("version", "1")]),
    ("limits", Json.mkObj [
      ("maxFrameBytes", maxFrameBytes),
      ("maxToolOutputBytes", 262_144),
      ("maxErrorMessageBytes", 4_096)
    ])
  ]
  let initialized ← readFrame child.stdout
  assert (← IO.ofExcept (responseOk initialized)) "Git plugin must initialize"

  writeFrame input <| request "tools_1" "list_tools" (Json.mkObj [])
  let listed ← readFrame child.stdout
  assert (← IO.ofExcept (responseOk listed)) "Git plugin must list tools"
  let result ← IO.ofExcept (listed.getObjVal? "result")
  let toolJson ← IO.ofExcept (result.getObjVal? "tools")
  let tools ← IO.ofExcept toolJson.getArr?
  assert (hasTool tools "git_read") "Git plugin must expose git_read at a repository root"
  assert (hasTool tools "git_write") "Git plugin must expose git_write at a repository root"

  writeFrame input <| request "call_1" "call_tool" <| Json.mkObj [
    ("name", "git_read"),
    ("arguments", Json.mkObj [
      ("operation", "status"),
      ("arguments", Json.arr ((#["--short"] : Array String).map toJson)),
      ("command", "git status --short")
    ])
  ]
  let called ← readFrame child.stdout
  assert (← IO.ofExcept (responseOk called)) "Git status must execute through the Lean plugin"
  let callResult ← IO.ofExcept (called.getObjVal? "result")
  let outputJson ← IO.ofExcept (callResult.getObjVal? "output")
  let output ← IO.ofExcept outputJson.getStr?
  assert (!output.isEmpty) "Git status output must be returned"

  writeFrame input <| request "shutdown_1" "shutdown" (Json.mkObj [])
  let shutdown ← readFrame child.stdout
  assert (← IO.ofExcept (responseOk shutdown)) "Git plugin must shut down cleanly"
  let exitCode ← child.wait
  assert (exitCode == 0) "Git plugin must exit successfully"

def main : IO Unit := do
  testGitPluginProtocol
  IO.println "Lean Git plugin tests passed"
