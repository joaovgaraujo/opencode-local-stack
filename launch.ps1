# launch.ps1 - open OpenCode Local in a project folder, in a new terminal.
# Used by the Start menu shortcut (asks for a folder) and by the Explorer
# right-click entry "Open with OpenCode Local" (gets the folder as argument).
param([string]$Folder)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms

$launcher = Join-Path $PSScriptRoot "opencode-local.cmd"
if (-not (Test-Path -LiteralPath $launcher)) {
    [System.Windows.Forms.MessageBox]::Show(
        "OpenCode Local is not set up yet. Run 'OpenCode Local - Change model or settings' " +
        "from the Start menu first.", "OpenCode Local") | Out-Null
    exit 1
}

if (-not $Folder) {
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.Description = "Choose the project folder to open in OpenCode"
    $dlg.SelectedPath = [Environment]::GetFolderPath("MyDocuments")
    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
    $Folder = $dlg.SelectedPath
}
# "D:\" style roots: a trailing backslash before a closing quote escapes it.
$Folder = $Folder.TrimEnd('\')
if ($Folder -match '^[A-Za-z]:$') { $Folder += '\.' }

$wt = Get-Command wt.exe -ErrorAction SilentlyContinue
if ($wt) {
    # Windows Terminal renders OpenCode's interface better than the old console.
    # Start-Process doesn't quote list items that contain spaces: the title
    # must be quoted by hand, or wt runs "Local" as the command.
    Start-Process $wt.Source -ArgumentList @("-d", "`"$Folder`"", "--title", "`"OpenCode Local`"",
                                             "cmd", "/k", "`"$launcher`"")
} else {
    Start-Process cmd.exe -WorkingDirectory $Folder -ArgumentList @("/k", "title OpenCode Local & `"$launcher`"")
}
