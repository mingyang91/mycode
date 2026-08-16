import MyCode.Protocol

open Lean

namespace MyCode

private def maxPlanChars : Nat := 65536
private def maxTodoPhases : Nat := 16
private def maxTodoTasks : Nat := 64
private def maxTodoTextChars : Nat := 200

private def stringField (json : Json) (name : String) : Except String String := do
  fromJson? (← json.getObjVal? name)

private def optionalStringField (json : Json) (name : String) : Except String (Option String) :=
  match json.getObjVal? name with
  | .ok value => some <$> fromJson? value
  | .error _ => pure none

private def stringArrayField (json : Json) (name : String) : Except String (Array String) := do
  fromJson? (← json.getObjVal? name)

private def requireSome (value : Option String) (message : String) : Except String String :=
  match value with
  | some content => pure content
  | none => throw message

private def validTodoStatus (status : String) : Bool :=
  ["pending", "in_progress", "completed", "abandoned", "blocked"].contains status

private def validateTodoText (kind text : String) : Except String Unit := do
  if text.trimAscii.isEmpty then throw s!"{kind} must not be empty"
  if text.length > maxTodoTextChars then throw s!"{kind} exceeds {maxTodoTextChars} characters"

private def validateTodoItem (item : TodoItem) : Except String Unit := do
  validateTodoText "todo task" item.content
  if !validTodoStatus item.status then throw s!"unknown todo status: {item.status}"
  match item.blocker? with
  | some blocker => validateTodoText "todo blocker" blocker
  | none => pure ()

public def validateTodoPhases (phases : Array TodoPhase) : Except String Unit := do
  if phases.size > maxTodoPhases then throw s!"todo list exceeds {maxTodoPhases} phases"
  let total := phases.foldl (fun count phase => count + phase.tasks.size) 0
  if total > maxTodoTasks then throw s!"todo list exceeds {maxTodoTasks} tasks"
  let mut phaseNames : Array String := #[]
  let mut taskNames : Array String := #[]
  for phase in phases do
    validateTodoText "todo phase" phase.name
    if phaseNames.contains phase.name then throw s!"duplicate todo phase: {phase.name}"
    phaseNames := phaseNames.push phase.name
    for task in phase.tasks do
      validateTodoItem task
      if taskNames.contains task.content then throw s!"duplicate todo task: {task.content}"
      taskNames := taskNames.push task.content

public def validatePlanContent (content : String) : Except String Unit := do
  if content.trimAscii.isEmpty then throw "plan content must not be empty"
  if content.length > maxPlanChars then throw s!"plan exceeds {maxPlanChars} characters"

private def taskExists (phases : Array TodoPhase) (content : String) : Bool :=
  phases.any (fun phase => phase.tasks.any (fun task => task.content == content))

private def phaseExists (phases : Array TodoPhase) (name : String) : Bool :=
  phases.any (fun phase => phase.name == name)

private def updateTaskStatus (task : TodoItem) (status : String) : TodoItem :=
  { task with status, blocker? := if status == "blocked" then task.blocker? else none }

private def setTaskStatus (phases : Array TodoPhase) (content status : String) : Except String (Array TodoPhase) := do
  if !taskExists phases content then throw s!"todo task not found: {content}"
  pure <| phases.map fun phase =>
    { phase with tasks := phase.tasks.map fun task =>
        if task.content == content then updateTaskStatus task status else task }

private def startTask (phases : Array TodoPhase) (content : String) : Except String (Array TodoPhase) := do
  if !taskExists phases content then throw s!"todo task not found: {content}"
  pure <| phases.map fun phase =>
    { phase with tasks := phase.tasks.map fun task =>
        if task.content == content then { task with status := "in_progress", blocker? := none }
        else if task.status == "in_progress" then { task with status := "pending" }
        else task }

private def setTargetStatus (phases : Array TodoPhase) (task? phase? : Option String)
    (status : String) : Except String (Array TodoPhase) := do
  match task?, phase? with
  | some task, none => setTaskStatus phases task status
  | none, some phase =>
    if !phaseExists phases phase then throw s!"todo phase not found: {phase}"
    pure <| phases.map fun current =>
      if current.name == phase then
        { current with tasks := current.tasks.map fun task => updateTaskStatus task status }
      else current
  | none, none =>
    pure <| phases.map fun phase =>
      { phase with tasks := phase.tasks.map fun task => updateTaskStatus task status }
  | some _, some _ => throw "todo operation accepts either task or phase, not both"

private def blockTarget (phases : Array TodoPhase) (task? phase? : Option String)
    (reason? : Option String) : Except String (Array TodoPhase) := do
  if task?.isNone && phase?.isNone then throw "todo block requires a task or phase"
  match reason? with
  | some reason => validateTodoText "todo blocker" reason
  | none => pure ()
  let updated ← setTargetStatus phases task? phase? "blocked"
  pure <| updated.map fun phase =>
    { phase with tasks := phase.tasks.map fun task =>
        let selectedTask := match task? with | some content => content == task.content | none => false
        let selectedPhase := match phase? with | some name => name == phase.name | none => false
        if (selectedTask || selectedPhase) && task.status == "blocked" then
          { task with blocker? := reason? }
        else task }

private def removeTarget (phases : Array TodoPhase) (task? phase? : Option String) : Except String (Array TodoPhase) := do
  match task?, phase? with
  | some task, none =>
    if !taskExists phases task then throw s!"todo task not found: {task}"
    pure <| phases.map fun current =>
      { current with tasks := current.tasks.filter (fun item => item.content != task) }
  | none, some phase =>
    if !phaseExists phases phase then throw s!"todo phase not found: {phase}"
    pure <| phases.filter (fun current => current.name != phase)
  | none, none => pure #[]
  | some _, some _ => throw "todo remove accepts either task or phase, not both"

private def appendTasks (phases : Array TodoPhase) (phaseName : String)
    (items : Array String) : Except String (Array TodoPhase) := do
  validateTodoText "todo phase" phaseName
  if items.isEmpty then throw "todo append requires at least one task"
  let mut next := phases
  if !phaseExists next phaseName then
    if next.size >= maxTodoPhases then throw s!"todo list exceeds {maxTodoPhases} phases"
    next := next.push { name := phaseName }
  for item in items do
    validateTodoText "todo task" item
    if taskExists next item then throw s!"duplicate todo task: {item}"
    next := next.map fun phase =>
      if phase.name == phaseName then { phase with tasks := phase.tasks.push { content := item } }
      else phase
  validateTodoPhases next
  pure next

private def todoSummary (operation : String) (phases : Array TodoPhase) : String :=
  let total := phases.foldl (fun count phase => count + phase.tasks.size) 0
  let remaining := phases.foldl (fun count phase =>
    count + phase.tasks.foldl (fun inner task =>
      if task.status == "completed" || task.status == "abandoned" then inner else inner + 1) 0) 0
  s!"Todo {operation} applied: {remaining} open of {total} tasks."

private def applyTodoCall (state : State) (arguments : Json) : Except String (State × String) := do
  let operation ← stringField arguments "op"
  let task? ← optionalStringField arguments "task"
  let phase? ← optionalStringField arguments "phase"
  let nextTodos ← match operation with
    | "init" =>
      let phases : Array TodoPhase ← fromJson? (← arguments.getObjVal? "phases")
      validateTodoPhases phases
      pure phases
    | "append" =>
      let phase ← requireSome phase? "todo append requires phase"
      appendTasks state.todos phase (← stringArrayField arguments "items")
    | "start" =>
      let task ← requireSome task? "todo start requires task"
      startTask state.todos task
    | "done" => setTargetStatus state.todos task? phase? "completed"
    | "drop" => setTargetStatus state.todos task? phase? "abandoned"
    | "block" => blockTarget state.todos task? phase? (← optionalStringField arguments "reason")
    | "unblock" => setTargetStatus state.todos task? phase? "pending"
    | "rm" => removeTarget state.todos task? phase?
    | "view" => pure state.todos
    | _ => throw s!"unknown todo operation: {operation}"
  validateTodoPhases nextTodos
  pure ({ state with todos := nextTodos }, todoSummary operation nextTodos)

private def applyPlanCall (state : State) (arguments : Json) : Except String (State × String × Bool) := do
  if !state.plan.enabled then throw "plan tool is available only in plan mode"
  let operation ← stringField arguments "op"
  let content ← stringField arguments "content"
  validatePlanContent content
  let review := operation == "propose"
  if operation != "update" && !review then throw s!"unknown plan operation: {operation}"
  let revision := state.plan.revision + 1
  let next := { state with plan := {
    enabled := true
    revision
    status := if review then "review" else "draft"
    content
  } }
  let summary := if review then s!"Plan revision {revision} submitted for review."
    else s!"Plan revision {revision} updated."
  pure (next, summary, review)

public structure BuiltinToolResult where
  state : State
  content : String
  isError : Bool := false
  requestPlanReview : Bool := false

public def applyBuiltinTool? (state : State) (call : ToolCall) : Option BuiltinToolResult :=
  if call.name == "todo" then
    some <| match applyTodoCall state call.arguments with
      | .ok (next, content) => { state := next, content }
      | .error error => { state, content := error, isError := true }
  else if call.name == "plan" then
    some <| match applyPlanCall state call.arguments with
      | .ok (next, content, review) => { state := next, content, requestPlanReview := review }
      | .error error => { state, content := error, isError := true }
  else none

end MyCode
