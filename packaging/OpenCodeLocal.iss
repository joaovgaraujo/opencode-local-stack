; OpenCodeLocal.iss - Inno Setup script for the OpenCode Local installer.
; Build with Build-Installer.ps1 (or: ISCC.exe /DMyAppVersion=1.1.0 packaging\OpenCodeLocal.iss)
;
; The installer copies the Python installer/launcher files, asks for a model,
; a context size and a thinking mode, then runs setup.cmd with those answers in
; a console window (it downloads Python if missing, Node.js + OpenCode,
; llama.cpp and the model, ~20-25 GB, and runs the tests).
;
; Silent / scripted install: OCLARGS replaces the page answers, e.g.
;   OpenCodeLocal-Setup.exe /VERYSILENT /DIR="D:\OpenCodeLocal" /OCLARGS="--model auto --think off"

#ifndef MyAppVersion
  #define MyAppVersion "1.1.1"
#endif
#define MyAppName "OpenCode Local"
#define MyOutputBase "OpenCodeLocal-Setup"
#define PS "{sys}\WindowsPowerShell\v1.0\powershell.exe"
; Quotes are doubled twice: once for the preprocessor, once for the section string.
#define LaunchArgs "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """"{app}\launch.ps1"""""

[Setup]
; Stable AppId so upgrades replace the existing install
AppId={{5C14FD04-C95F-42F6-88EA-3C585429236E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=joaovgaraujo
AppPublisherURL=https://github.com/joaovgaraujo/opencode-local-stack
AppSupportURL=https://github.com/joaovgaraujo/opencode-local-stack/issues
; Per-user install, no admin prompt: everything it sets up (Node packages,
; OpenCode config, the PATH launcher) is per-user too. With lowest privileges
; {autopf} is %LOCALAPPDATA%\Programs.
PrivilegesRequired=lowest
DefaultDirName={autopf}\OpenCodeLocal
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UsePreviousAppDir=yes
DirExistsWarning=no
; The model and llama.cpp land in the install folder: ~20-25 GB for the
; default model. Shown on the folder page so people pick a drive with room.
ExtraDiskSpaceRequired=25000000000
OutputDir=..\dist
OutputBaseFilename={#MyOutputBase}
SetupIconFile=..\assets\opencode-local.ico
UninstallDisplayIcon={app}\assets\opencode-local.ico
UninstallDisplayName={#MyAppName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; winget, Windows Terminal and the CUDA builds assume Windows 10 or later.
MinVersion=10.0
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany=joaovgaraujo
VersionInfoProductName={#MyAppName}
VersionInfoDescription={#MyAppName} Setup
VersionInfoCopyright=Copyright (C) 2026 joaovgaraujo
VersionInfoOriginalFileName={#MyOutputBase}.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[CustomMessages]
english.ModelCaption=Model
english.ModelDesc=Which model should run on this PC?
english.ModelSub=Models are matched to your total GPU memory and RAM. All of them run offline once downloaded.
english.ModelAuto=Recommended for this PC (picked from your GPU and RAM)
english.ModelLink=A GGUF model from a Hugging Face link
english.LinkCaption=Hugging Face model
english.LinkDesc=Paste the link to a GGUF model repository.
english.LinkSub=For example https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF. A link to a single .gguf file picks that file; otherwise the Q4_K_M-size file is used.
english.LinkPrompt=Link:
english.LinkInvalid=Please paste a Hugging Face link such as https://huggingface.co/OWNER/NAME.
english.CtxCaption=Context size
english.CtxDesc=How much code and conversation the model sees at once.
english.CtxSub=Automatic uses 64K tokens below 16 GB of GPU memory, 128K below 24 GB and 256K above. A smaller context leaves more GPU memory for the model, so answers come faster.
english.CtxAuto=Automatic (recommended)
english.ThinkCaption=Thinking
english.ThinkDesc=Should the model reason before it answers?
english.ThinkSub=You can change this any time: opencode-local --think off, limited or on.
english.ThinkLimited=Limited: thinks up to ~4,000 tokens, about a minute at most (recommended)
english.ThinkOff=Off: answers right away; fastest, a bit weaker on hard bugs
english.ThinkOn=Unlimited: best on hard problems, can take several minutes
english.TaskGroup=Options:
english.TaskScripts=Let PowerShell run scripts for my user (RemoteSigned), so typing "opencode" works
english.TaskTests=Test the model and OpenCode after setup (a few minutes)
english.TaskContext=Add "Open with OpenCode Local" to the folder right-click menu
english.ContextMenu=Open with OpenCode Local
english.RunningSetup=Setting up the model, llama.cpp and OpenCode in the console window. The download can take a while...
english.SetupFailed=The model setup did not finish. The console window shows why (it is also in setup.log in the install folder). Fix the problem and run "OpenCode Local - Change model or settings" from the Start menu.
english.FinishedOK=OpenCode Local is ready.%n%nTo code: right-click a project folder and choose "Open with OpenCode Local" (on Windows 11 it is under "Show more options"), use the Start menu entry, or type  opencode-local  in a terminal inside the folder.
english.FinishedFail=The files are installed, but the model setup did not finish. Run "OpenCode Local - Change model or settings" from the Start menu to try again.
english.LaunchNow=Open a project in OpenCode now
english.StartSettings=OpenCode Local - Change model or settings
english.StartStop=OpenCode Local - Stop model server
english.DeleteModels=Also delete the downloaded models and llama.cpp? They can be 20 GB or more.%n%nChoose No to keep them, for example to reinstall later without downloading again.

brazilianportuguese.ModelCaption=Modelo
brazilianportuguese.ModelDesc=Qual modelo deve rodar neste PC?
brazilianportuguese.ModelSub=Os modelos são escolhidos pela memória total da placa de vídeo e da RAM. Depois de baixados, todos funcionam offline.
brazilianportuguese.ModelAuto=Recomendado para este PC (escolhido pela placa de vídeo e RAM)
brazilianportuguese.ModelLink=Um modelo GGUF por um link do Hugging Face
brazilianportuguese.LinkCaption=Modelo do Hugging Face
brazilianportuguese.LinkDesc=Cole o link de um repositório de modelo GGUF.
brazilianportuguese.LinkSub=Por exemplo https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF. Um link para um único arquivo .gguf usa esse arquivo; senão é usado o arquivo do tamanho Q4_K_M.
brazilianportuguese.LinkPrompt=Link:
brazilianportuguese.LinkInvalid=Cole um link do Hugging Face, como https://huggingface.co/DONO/NOME.
brazilianportuguese.CtxCaption=Tamanho do contexto
brazilianportuguese.CtxDesc=Quanto código e conversa o modelo enxerga de uma vez.
brazilianportuguese.CtxSub=Automático usa 64K tokens com menos de 16 GB na placa de vídeo, 128K com menos de 24 GB e 256K acima disso. Um contexto menor deixa mais memória da placa para o modelo, e as respostas saem mais rápido.
brazilianportuguese.CtxAuto=Automático (recomendado)
brazilianportuguese.ThinkCaption=Raciocínio
brazilianportuguese.ThinkDesc=O modelo deve pensar antes de responder?
brazilianportuguese.ThinkSub=Dá para mudar quando quiser: opencode-local --think off, limited ou on.
brazilianportuguese.ThinkLimited=Limitado: pensa até ~4.000 tokens, no máximo cerca de um minuto (recomendado)
brazilianportuguese.ThinkOff=Desligado: responde na hora; mais rápido, um pouco pior em bugs difíceis
brazilianportuguese.ThinkOn=Sem limite: melhor em problemas difíceis, pode levar vários minutos
brazilianportuguese.TaskGroup=Opções:
brazilianportuguese.TaskScripts=Permitir que o PowerShell rode scripts no meu usuário (RemoteSigned), para o comando "opencode" funcionar
brazilianportuguese.TaskTests=Testar o modelo e o OpenCode depois da instalação (alguns minutos)
brazilianportuguese.TaskContext=Adicionar "Abrir com OpenCode Local" ao menu do botão direito das pastas
brazilianportuguese.ContextMenu=Abrir com OpenCode Local
brazilianportuguese.RunningSetup=Instalando o modelo, o llama.cpp e o OpenCode na janela do console. O download pode demorar...
brazilianportuguese.SetupFailed=A instalação do modelo não terminou. A janela do console mostra o motivo (também está em setup.log na pasta de instalação). Corrija o problema e rode "OpenCode Local - Trocar modelo ou configurações" no menu Iniciar.
brazilianportuguese.FinishedOK=O OpenCode Local está pronto.%n%nPara programar: clique com o botão direito numa pasta de projeto e escolha "Abrir com OpenCode Local" (no Windows 11 fica em "Mostrar mais opções"), use o atalho do menu Iniciar ou digite  opencode-local  num terminal dentro da pasta.
brazilianportuguese.FinishedFail=Os arquivos foram instalados, mas a instalação do modelo não terminou. Rode "OpenCode Local - Trocar modelo ou configurações" no menu Iniciar para tentar de novo.
brazilianportuguese.LaunchNow=Abrir um projeto no OpenCode agora
brazilianportuguese.StartSettings=OpenCode Local - Trocar modelo ou configurações
brazilianportuguese.StartStop=OpenCode Local - Parar o servidor do modelo
brazilianportuguese.DeleteModels=Apagar também os modelos baixados e o llama.cpp? Podem ocupar 20 GB ou mais.%n%nEscolha Não para mantê-los, por exemplo para reinstalar depois sem baixar de novo.

[Tasks]
Name: "allowscripts"; Description: "{cm:TaskScripts}"; GroupDescription: "{cm:TaskGroup}"
Name: "runtests"; Description: "{cm:TaskTests}"; GroupDescription: "{cm:TaskGroup}"
Name: "contextmenu"; Description: "{cm:TaskContext}"; GroupDescription: "{cm:TaskGroup}"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\install.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\start.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\setup.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\setup.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\launch.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
; custom.py is local-only tooling that never ships
Source: "..\installer\*.py"; DestDir: "{app}\installer"; Excludes: "custom.py"; Flags: ignoreversion
Source: "..\tests\*.py"; DestDir: "{app}\tests"; Flags: ignoreversion
Source: "..\tests\*.ps1"; DestDir: "{app}\tests"; Flags: ignoreversion
Source: "..\docs\*.md"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\models\PUT_GGUF_HERE.txt"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "..\assets\opencode-local.ico"; DestDir: "{app}\assets"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{#PS}"; Parameters: "{#LaunchArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\assets\opencode-local.ico"
Name: "{group}\{cm:StartSettings}"; Filename: "{app}\setup.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\assets\opencode-local.ico"
Name: "{group}\{cm:StartStop}"; Filename: "{app}\opencode-local.cmd"; Parameters: "--stop"; WorkingDir: "{app}"; IconFilename: "{app}\assets\opencode-local.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{#PS}"; Parameters: "{#LaunchArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\assets\opencode-local.ico"; Tasks: desktopicon

[Registry]
; Right-click on a folder, and on the empty space inside one (Directory\Background)
Root: HKCU; Subkey: "Software\Classes\Directory\shell\OpenCodeLocal"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenu}"; Flags: uninsdeletekey; Tasks: contextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\OpenCodeLocal"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\assets\opencode-local.ico"; Tasks: contextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\shell\OpenCodeLocal\command"; ValueType: string; ValueName: ""; ValueData: """{#PS}"" {#LaunchArgs} ""%V"""; Tasks: contextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\OpenCodeLocal"; ValueType: string; ValueName: ""; ValueData: "{cm:ContextMenu}"; Flags: uninsdeletekey; Tasks: contextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\OpenCodeLocal"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\assets\opencode-local.ico"; Tasks: contextmenu
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\OpenCodeLocal\command"; ValueType: string; ValueName: ""; ValueData: """{#PS}"" {#LaunchArgs} ""%V"""; Tasks: contextmenu

[Run]
Filename: "{#PS}"; Parameters: "{#LaunchArgs}"; Description: "{cm:LaunchNow}"; Flags: postinstall nowait skipifsilent; Check: SetupSucceeded

[UninstallDelete]
; Files the installer and opencode-local create next to themselves. models\
; and llama.cpp\ are handled in [Code] (the uninstaller asks first).
Type: files; Name: "{app}\settings.json"
Type: files; Name: "{app}\.server-state.json"
Type: files; Name: "{app}\.llama-server.pid"
Type: files; Name: "{app}\server.log"
Type: files; Name: "{app}\server.err"
Type: files; Name: "{app}\setup.log"
Type: files; Name: "{app}\run.ps1"
Type: files; Name: "{app}\run.sh"
Type: files; Name: "{app}\opencode.json"
Type: files; Name: "{app}\opencode-local.cmd"
Type: files; Name: "{app}\RESULTS.md"
Type: filesandordirs; Name: "{app}\opencode-scratch"
Type: filesandordirs; Name: "{app}\installer\__pycache__"
Type: filesandordirs; Name: "{app}\tests\__pycache__"
Type: filesandordirs; Name: "{app}\llama.cpp.new"
Type: filesandordirs; Name: "{app}\llama.cpp.old"
Type: dirifempty; Name: "{app}\models"
Type: dirifempty; Name: "{app}"

[Code]
var
  ModelPage, CtxPage, ThinkPage: TInputOptionWizardPage;
  LinkPage: TInputQueryWizardPage;
  SetupOK: Boolean;

procedure InitializeWizard;
begin
  ModelPage := CreateInputOptionPage(wpSelectDir, CustomMessage('ModelCaption'),
    CustomMessage('ModelDesc'), CustomMessage('ModelSub'), True, False);
  ModelPage.Add(CustomMessage('ModelAuto'));
  ModelPage.Add(CustomMessage('ModelLink'));
  ModelPage.SelectedValueIndex := 0;

  LinkPage := CreateInputQueryPage(ModelPage.ID, CustomMessage('LinkCaption'),
    CustomMessage('LinkDesc'), CustomMessage('LinkSub'));
  LinkPage.Add(CustomMessage('LinkPrompt'), False);

  CtxPage := CreateInputOptionPage(LinkPage.ID, CustomMessage('CtxCaption'),
    CustomMessage('CtxDesc'), CustomMessage('CtxSub'), True, False);
  CtxPage.Add(CustomMessage('CtxAuto'));
  CtxPage.Add('32K tokens');
  CtxPage.Add('64K tokens');
  CtxPage.Add('128K tokens');
  CtxPage.Add('256K tokens');
  CtxPage.SelectedValueIndex := 0;

  ThinkPage := CreateInputOptionPage(CtxPage.ID, CustomMessage('ThinkCaption'),
    CustomMessage('ThinkDesc'), CustomMessage('ThinkSub'), True, False);
  ThinkPage.Add(CustomMessage('ThinkLimited'));
  ThinkPage.Add(CustomMessage('ThinkOff'));
  ThinkPage.Add(CustomMessage('ThinkOn'));
  ThinkPage.SelectedValueIndex := 0;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = LinkPage.ID) and (ModelPage.SelectedValueIndex <> 1);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Link: String;
begin
  Result := True;
  if CurPageID = LinkPage.ID then
  begin
    Link := Trim(LinkPage.Values[0]);
    if (Pos('huggingface.co/', Lowercase(Link)) = 0) and (Pos('hf.co/', Lowercase(Link)) = 0) then
    begin
      MsgBox(CustomMessage('LinkInvalid'), mbError, MB_OK);
      Result := False;
    end;
  end;
end;

function SetupArgs: String;
var
  Extra: String;
begin
  Result := '--non-interactive --update-llamacpp --pause-on-error --log "' +
            ExpandConstant('{app}\setup.log') + '"';
  Extra := ExpandConstant('{param:OCLARGS|}');
  if Extra <> '' then
  begin
    Result := Result + ' ' + Extra;
    exit;
  end;
  case ModelPage.SelectedValueIndex of
    0: Result := Result + ' --model auto';
    1: Result := Result + ' --hf ' + Trim(LinkPage.Values[0]);
  end;
  case CtxPage.SelectedValueIndex of
    1: Result := Result + ' --ctx 32768';
    2: Result := Result + ' --ctx 65536';
    3: Result := Result + ' --ctx 131072';
    4: Result := Result + ' --ctx 262144';
  end;
  case ThinkPage.SelectedValueIndex of
    0: Result := Result + ' --think limited';
    1: Result := Result + ' --think off';
    2: Result := Result + ' --think on';
  end;
  if not WizardIsTaskSelected('runtests') then
    Result := Result + ' --skip-tests';
  if WizardIsTaskSelected('allowscripts') then
    Result := Result + ' --allow-ps-scripts';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  RC: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    WizardForm.StatusLabel.Caption := CustomMessage('RunningSetup');
    SetupOK := Exec(ExpandConstant('{cmd}'),
      '/c ""' + ExpandConstant('{app}\setup.cmd') + '" ' + SetupArgs + '"',
      ExpandConstant('{app}'), SW_SHOWNORMAL, ewWaitUntilTerminated, RC) and (RC = 0);
    if not SetupOK then
      SuppressibleMsgBox(CustomMessage('SetupFailed'), mbError, MB_OK, IDOK);
  end;
end;

function SetupSucceeded: Boolean;
begin
  Result := SetupOK;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
  begin
    if SetupOK then
      WizardForm.FinishedLabel.Caption := CustomMessage('FinishedOK')
    else
      WizardForm.FinishedLabel.Caption := CustomMessage('FinishedFail');
  end;
end;

// Remove an opencode-local.cmd launcher on PATH only if it points at this install.
procedure RemoveLauncher(Path: String);
var
  Body: AnsiString;
begin
  if FileExists(Path) and LoadStringFromFile(Path, Body) and
     (Pos(Lowercase(ExpandConstant('{app}')), Lowercase(String(Body))) > 0) then
    DeleteFile(Path);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  RC: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    if FileExists(ExpandConstant('{app}\opencode-local.cmd')) then
      Exec(ExpandConstant('{cmd}'), '/c ""' + ExpandConstant('{app}\opencode-local.cmd') + '" --stop"',
           ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, RC);
    RemoveLauncher(ExpandConstant('{userappdata}\npm\opencode-local.cmd'));
    RemoveLauncher(ExpandConstant('{localappdata}\Microsoft\WindowsApps\opencode-local.cmd'));
    if (DirExists(ExpandConstant('{app}\models')) or DirExists(ExpandConstant('{app}\llama.cpp'))) and
       (SuppressibleMsgBox(CustomMessage('DeleteModels'), mbConfirmation, MB_YESNO, IDNO) = IDYES) then
    begin
      DelTree(ExpandConstant('{app}\models'), True, True, True);
      DelTree(ExpandConstant('{app}\llama.cpp'), True, True, True);
    end;
  end;
end;
