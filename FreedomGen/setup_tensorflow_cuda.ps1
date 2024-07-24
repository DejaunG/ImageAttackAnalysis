# PowerShell script to automate the setup of TensorFlow with CUDA support

# Define the versions
$tensorflowVersion = "2.17.0"
$cudaVersion = "12.0"
$cudnnVersion = "8.9.7"

# Corrected download URLs
$cudaURL = "https://developer.download.nvidia.com/compute/cuda/12.0/local_installers/cuda_12.0.0_520.61.05_windows.exe"
$cudnnURL = "https://developer.download.nvidia.com/compute/redist/cudnn/v8.9.7/cudnn-windows-x86_64-8.9.7.15.zip"

# Define paths
$cudaInstallerPath = "$env:TEMP\cuda_installer.exe"
$cudnnZipPath = "$env:TEMP\cudnn.zip"
$cudnnExtractPath = "$env:TEMP\cudnn"

# Download and install CUDA
Write-Output "Downloading CUDA $cudaVersion..."
Invoke-WebRequest -Uri $cudaURL -OutFile $cudaInstallerPath

Write-Output "Installing CUDA $cudaVersion..."
Start-Process -FilePath $cudaInstallerPath -ArgumentList "/silent", "/noreboot" -NoNewWindow -Wait

# Download and extract cuDNN
Write-Output "Downloading cuDNN $cudnnVersion..."
Invoke-WebRequest -Uri $cudnnURL -OutFile $cudnnZipPath

Write-Output "Extracting cuDNN $cudnnVersion..."
Expand-Archive -Path $cudnnZipPath -DestinationPath $cudnnExtractPath

# Copy cuDNN files to CUDA directory
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVersion"
Copy-Item -Path "$cudnnExtractPath\cuda\bin\*" -Destination "$cudaPath\bin" -Force
Copy-Item -Path "$cudnnExtractPath\cuda\include\*" -Destination "$cudaPath\include" -Force
Copy-Item -Path "$cudnnExtractPath\cuda\lib\x64\*" -Destination "$cudaPath\lib\x64" -Force

# Install TensorFlow
Write-Output "Installing TensorFlow $tensorflowVersion..."
pip install tensorflow==$tensorflowVersion

Write-Output "Setup complete!"
