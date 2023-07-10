# Welcome to the CUDA Skin Detector

## This project is both an implementation of a CUDA Skin detector, and a short guide on how to configure a CMake CUDA project.

It covers the basis for creating a functional library that uses CUDA and OpenCV. In later stages, we show how to use this library in an executable. The project takes advantage of modern CMake syntax and CUDA.

## Building the project

Start by clonning this repository from [https://github.com/LuisTbx/SkinCuda](https://github.com/LuisTbx/SkinCuda)

Before compiling, create a `test_images` directory and place some images containing people with different skin colors. 

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

At this point we are ready to buld the project, use `make` to start building. Once build you will find two executables, `skin` and `skinMask`. When ran they will produce `map.jpeg` and `mask.jpg`

```
make -j6
```

And that is it, if you have any suggestion please reach out by crating an issue or a PR!