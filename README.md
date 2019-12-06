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
   
  ## alternatively 
  1. complete steps above until 2.
  2. Configure Visual Studio
	1. install opencv in C:\ opencv\build\x64\vc15\bin 
		1. if this is not where you installed it then change the directory in the property sheet OpenCV_customCuda_v1.prop 
	2. Go to View -> Other Windows -> Property Manager. (May need to turn on expert settings.)
	3. Click the Add Existing Project Property Sheet icon in Property Manager and add the downloaded property sheet.
  3. Run test program openCV_Cuda_TestProgram.cpp to test and check linking

  
