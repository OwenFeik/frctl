build: *.cpp
	g++ frctl.cpp \
		-o bin/frctl \
		-I /usr/include/opencv4 \
		-lopencv_core \
		-lopencv_imgcodecs

run: build
	./bin/frctl
