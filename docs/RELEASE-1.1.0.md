A proper Windows installer: download **OpenCodeLocal-Setup.exe**, run it, and answer a few pages. No git, Python or admin rights needed beforehand.

## Install

1. Download **OpenCodeLocal-Setup.exe** below and run it. The installer isn't code-signed, so Windows may show "Windows protected your PC": choose **More info**, then **Run anyway**.
2. Pick an install folder on a drive with about 25 GB free (the model and llama.cpp go there).
3. Choose:
   - **Model:** recommended for this PC (picked from total GPU memory and RAM), or a GGUF model from a Hugging Face link.
   - **Context:** automatic (64K tokens below 16 GB of GPU memory, 128K below 24 GB, 256K above) or a fixed size.
   - **Thinking:** limited to about 4,000 tokens (default), off, or unlimited.
   - **Options:**
     - let PowerShell run scripts for your user
     - test the model after setup
     - add "Open with OpenCode Local" to the folder right-click menu
     - desktop icon
4. A console window then installs Python (if missing), Node.js, OpenCode, the newest llama.cpp for your GPU driver and the model, and runs the tests. Its output also goes to `setup.log` in the install folder.

## Use

- Right-click a project folder and choose **Open with OpenCode Local**. On Windows 11 it is under **Show more options**.
- Or use the **OpenCode Local** Start menu entry, which asks for a folder.
- Or type `opencode-local` in a terminal inside the folder.

The Start menu also has **Change model or settings**, which reruns the setup wizard, and **Stop model server**. Uninstalling asks whether to delete the downloaded models, so you can keep them for a reinstall.

## Changes since 1.0.0

- Windows installer built with Inno Setup, in English and Brazilian Portuguese, installed per user.
- `install.py --model auto` installs the catalog model that best fits the PC.
- Download progress is shown while the model downloads.
- `--update-llamacpp` skips the download when the installed build is already the newest.
- `--log FILE` copies the installer output to a file.
- Fixed: arguments given to `setup.cmd` (such as `--hf`) were taken by Windows PowerShell as its own parameters and never reached the installer.

`opencode-local-installer.cmd`, the script installer from 1.0.0, is still attached for anyone who prefers it.
