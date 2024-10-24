# BUILD
FROM alpine:3.18
RUN set -ex && \
	apk --no-cache --update add \
    cmake g++ gcc git linux-headers libpthread-stubs make 
WORKDIR /flexfringe
COPY . ./
RUN mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j$(nproc)
RUN build/runtests

# RUN
FROM alpine:3.18
RUN set -ex && \
	apk --no-cache --update add \
    libstdc++
RUN addgroup -S flexfringe && adduser -S flexfringe -G flexfringe
USER flexfringe
WORKDIR /home/flexfringe
COPY --from=0 /flexfringe/build/flexfringe .
COPY ini ./ini
ENTRYPOINT ["./flexfringe"]
