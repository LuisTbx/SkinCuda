# Welcome to the CUDA Skin Detector

## This project is both an implementation of a CUDA Skin detector, and a short example on how to configure a CMake CUDA project.

It covers the basis for creating a functional library that uses CUDA and OpenCV. In later stages, we show how to use this library in an executable. The project takes advantage of modern CMake syntax and CUDA.

## Building the project

Start by clonning this repository from [https://github.com/LuisTbx/SkinCuda](https://github.com/LuisTbx/SkinCuda)

Before compiling, create a `test_images` directory and place some images containing people with different skin colors.  You can also download some videos such as: 


```
mkdir test_images
cd test_images
wget -O test_images/test_image.jpg https://assets.weforum.org/article/image/XaHpf_z51huQS_JPHs-jkPhBp0dLlxFJwt-sPLpGJB0.jpg
cd ../
```

We use CMake to build this project.  Let's create now a `build` directory, then run `cmake`

```
mkdir build
cd build
cmake ../
```

You may find issues about the GPU architecture, please adjust accordingly on lines 30, 44, and 58 of the `CMakeLists.txt` file. As it is it is mean to work with the 10XX `Pascal` series compute capability 61. 

```
set_target_properties(${SUBPROJECT} PROPERTIES CUDA_ARCHITECTURES "61")
```


At this point we are ready to buld the project, use `make` to start building. Once built you will find three executables, `skin` processes images from the webcam or vide, `skinMap` crates the skin map on a single image and `skinMask` produces the binay mask. After running the executables will produce `skinMap.jpg` and `skinMask.jpg` images.

```
make -j6
./skin # Process webcam
./skin test_images/video.avi # Process video file
./skinMap test_images/test_image.jpg
./skinMask test_images/test_image.jpg
ls
```
The `skinnMap` image is the result of applying the algorithm to the original image, it removes the background by blacking out the values (0) on the original image. The `skinMask` is a binary mask where the skin pixels are set to 255 and the background 0.


And that is it, hope you manage to reproduce these steps. If you have any suggestion or issue please reach out by crating an issue or a PR on GitHub!