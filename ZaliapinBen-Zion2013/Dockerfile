from debian:8.11

run apt-get update && apt-get install -y \
      clang \
      libgeographic-dev \
&& apt-get clean \
&& rm -fr /var/lib/apt/lists


copy zaliapin_ben_zion_2013.cpp build/
run cd build \
&& clang++ \
      -Wall \
      -O3 \
      -march=native \
      -std=c++14 \
      -lGeographic \
      -o zaliapin_ben_zion_2013.exe \
      zaliapin_ben_zion_2013.cpp \
&& mv zaliapin_ben_zion_2013.exe .. \
&& cd .. \
&& rm -fr build

entrypoint ["./zaliapin_ben_zion_2013.exe"]
