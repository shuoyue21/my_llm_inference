FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:13.1.0-devel-ubuntu24.04

# 可选：HTTP/HTTPS 代理（build 时通过 --build-arg 传入）
# ARG HTTP_PROXY
# ARG HTTPS_PROXY

# # 同时设置大小写，兼容 apt / git / wget 等
# ENV HTTP_PROXY=${HTTP_PROXY} \
#   HTTPS_PROXY=${HTTPS_PROXY} \
#   http_proxy=${HTTP_PROXY} \
#   https_proxy=${HTTPS_PROXY}

# 使用清华 TUNA Ubuntu 源 + 安装开发环境 & Ninja
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g; s|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  ninja-build \
  git \
  gfortran \
  wget \
  liblapack-dev \
  libarpack2-dev \
  libsuperlu-dev \
  && rm -rf /var/lib/apt/lists/*

# 你已经配置好的 GitHub 代理，不改
#RUN git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"

# 用 CMake + Ninja 编译并安装依赖
RUN set -eux; \
  cd /tmp; \
  \
  #### OpenBLAS
  git clone --depth=1 https://github.com/xianyi/OpenBLAS.git; \
  cmake -S OpenBLAS -B OpenBLAS/build -G Ninja -DUSE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release; \
  cmake --build OpenBLAS/build -- -j"$(nproc)"; \
  cmake --install OpenBLAS/build; \
  rm -rf OpenBLAS; \
  \
  #### Armadillo
  wget https://sourceforge.net/projects/arma/files/armadillo-15.2.1.tar.xz; \
  tar -xf armadillo-15.2.1.tar.xz; \
  cmake -S armadillo-15.2.1 -B armadillo-15.2.1/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SMOKE_TEST=OFF; \
  cmake --build armadillo-15.2.1/build -- -j"$(nproc)"; \
  cmake --install armadillo-15.2.1/build; \
  rm -rf armadillo-15.2.1*; \
  \
  #### googletest
  git clone --depth=1 https://github.com/google/googletest.git; \
  cmake -S googletest -B googletest/build -G Ninja -DCMAKE_BUILD_TYPE=Release; \
  cmake --build googletest/build -- -j"$(nproc)"; \
  cmake --install googletest/build; \
  rm -rf googletest; \
  \
  #### google benchmark
  git clone --depth=1 https://github.com/google/benchmark.git; \
  cmake -S benchmark -B benchmark/build -G Ninja \
  -DBENCHMARK_ENABLE_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release; \
  cmake --build benchmark/build -- -j"$(nproc)"; \
  cmake --install benchmark/build; \
  rm -rf benchmark; \
  \
  #### glog
  git clone --depth=1 https://github.com/google/glog.git; \
  cmake -S glog -B glog/build -G Ninja -DCMAKE_BUILD_TYPE=Release; \
  cmake --build glog/build -- -j"$(nproc)"; \
  cmake --install glog/build; \
  rm -rf glog; \
  \
  #### sentencepiece
  git clone --depth=1 https://github.com/google/sentencepiece.git; \
  cmake -S sentencepiece -B sentencepiece/build -G Ninja -DCMAKE_BUILD_TYPE=Release; \
  cmake --build sentencepiece/build -- -j"$(nproc)"; \
  cmake --install sentencepiece/build; \
  rm -rf sentencepiece

# 让 /usr/local/lib 更容易被找到（可选，但一般有用）
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# 以后你要调试的项目挂载到 /app
WORKDIR /app
