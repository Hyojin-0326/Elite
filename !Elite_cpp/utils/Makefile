CXX = g++
CXXFLAGS = -std=c++17 -O2 -fdiagnostics-color=always -g -shared -fPIC

# Open3D 헤더 및 라이브러리 경로
OPEN3D_INCLUDE = /home/hjkwon/dev/Open3D/cpp
EIGEN_INCLUDE = /home/hjkwon/dev/Open3D/3rdparty/Eigen
FMT_INCLUDE = /usr/include
OPEN3D_LIB = /usr/local/lib
OPEN3D_LIBNAME = Open3D

# Python & pybind11
PYTHON = python3.10
PYTHON_INC = $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT = $(shell $(PYTHON)-config --extension-suffix)

INCLUDES = -I$(OPEN3D_INCLUDE) -I$(EIGEN_INCLUDE) -I$(FMT_INCLUDE) $(PYTHON_INC)
LDFLAGS = -L$(OPEN3D_LIB) -l$(OPEN3D_LIBNAME)

SRC = session.cpp
TARGET = session$(PYTHON_EXT)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
