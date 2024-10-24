# BUILD
FROM alpine:3.18
RUN set -ex && \
	apk --no-cache --update add \
    sudo libstdc++ cmake g++ gcc bash git linux-headers libpthread-stubs make libpq python3-dev
WORKDIR /flexfringe
COPY . ./
RUN mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . -j$(nproc)
RUN build/runtests

# Make this usefull
ENV USER=flexfringe
ENV GROUPNAME=$USER
ENV UID=12345
ENV GID=23456
RUN addgroup \
    --gid "$GID" \
    "$GROUPNAME" \
&&  adduser \
    --disabled-password \
    --gecos "" \
    --home "$(pwd)" \
    --shell bash \
    --ingroup "$GROUPNAME" \
    --no-create-home \
    --uid "$UID" \
    $USER
RUN echo "$USER ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

WORKDIR /home/$USER
COPY . .
RUN cp /flexfringe/build/flexfringe .
USER $USER
ENTRYPOINT ["bash"]
