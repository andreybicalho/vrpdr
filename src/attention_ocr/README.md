# attention-ocr
A pytorch implementation of attention based ocr

This repo is still under development.

Inspired by the tensorflow attention ocr created by google. [link](https://github.com/tensorflow/models/tree/master/research/attention_ocr)

More details can also be found in this paper:

["Attention-based Extraction of Structured Information from Street View Imagery"](https://arxiv.org/abs/1704.03549)

# Install and Requirements

### pycrypto for Python 3.6, Windows 10, Visual Studio 2017:

1. open "x86_x64 Cross-Tools Command Prompt for VS 2017" with administrator privilege in start menu.
2. go to C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC and check your MSVC version (mine was 14.16.27023)
3. type set CL=-FI"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\include\stdint.h" with the version you've just found
(also typed it in the vscode env terminal and in the x86_x64 Cross-Tools Command Prompt for VS 2017 as well...)
4. simply pip install pycrypto
5. No module named 'winrandom' when using pycrypto: 
Problem is solved by editing string in crypto\Random\OSRNG\nt.py:
````
import winrandom
````
to
````
from . import winrandom
````