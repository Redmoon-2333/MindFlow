' Hidden autostart entry point for the MindFlow backend.
' The Startup folder URL points here; keep this filename stable.
Option Explicit
Dim fso, scriptDir, ps1, shell, args
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1 = scriptDir & "\start_mindflow_bg.ps1"
If Not fso.FileExists(ps1) Then
    WScript.Echo "Missing: " & ps1
    WScript.Quit 1
End If
Set shell = CreateObject("WScript.Shell")
args = ""
If WScript.Arguments.Count > 0 Then
    If WScript.Arguments(0) = "--open-ui" Then args = " -OpenUi"
End If
shell.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """" & args, 0, False
