# Lane Detection
Project utilizing OpenCV and CUDA to detect lanes using the Hough Transform.

## Project Setup
1. Get OpenCV
   1. For now, download the precompiled binaries [here](https://opencv.org/releases/).
   2. Extract and put in a good directory like `C:\OpenCV\`
   3. Add `<your opencv dir>\build\x64\vc15\bin` to the system PATH environment variable list.
1. Clone the Project
   1. `git clone https://github.com/WillTUW/Line-Detection.git`
2. Setup Visual Studio
   1. Add the OpenCV Include Directory to VS.
      1. Project -> Properties -> CUDA C/C++ -> Additional Include Directories -> Add `<your opencv dir>\build\include`
   2. Link OpenCV 
      1. Project -> Properties -> Linker -> Additional Include Directorie -> Add `<your opencv dir>\build\x64\vc15\lib`
      2. Project -> Properties -> Linker -> Input -> Additional Dependencies -> Add `opencv_world412d.lib`
         - The `opencv_world412d.lib` is for debugging, `opencv_world412.lib` is for release.
   
