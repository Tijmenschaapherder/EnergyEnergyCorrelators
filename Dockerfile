FROM quay.io/pypa/manylinux2010_x86_64
LABEL org.opencontainers.image.source=https://github.com/pkomiske/EnergyEnergyCorrelators
COPY scripts/prepare-eec-docker.sh /
RUN bash prepare-eec-docker.sh && rm prepare-eec-docker.sh
CMD ["/bin/bash"]