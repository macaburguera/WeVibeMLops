### Setting Up the Environment

1. **Create a Conda Environment**  
   ```bash
   conda create --name dogs python=3.10
   conda activate dogs
   ```
   This project has been tested with PyTorch 2.2.0 and CUDA 11.8.
2. **Install the Required Packages**

   First, install PyTorch manually according to your cuda version. For example, for CUDA 11.8:  

   ```bash
   pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
   ```
   Then go on with the rest. To install the package:

   ```bash
   pip install -e .
   ```
   Or either just the strict requirements:
   ```bash
   pip install -r requirements.txt
   ```
---
