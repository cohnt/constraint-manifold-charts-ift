# Single image that reproduces all three experiments of
#   "Planning along Differentiable Charts of Constraint Manifolds
#    with General-Purpose IK Solvers"
#
# Build:
#   git clone https://github.com/cohnt/constraint-manifold-charts-ift.git
#   cd constraint-manifold-charts-ift
#   docker build -t cohnt/constraint-manifold-charts-ift .
# Or pull the published image instead of building:
#   docker pull cohnt/constraint-manifold-charts-ift:v1
# Run:
#   docker run --rm -it cohnt/constraint-manifold-charts-ift
#
# The image is built from the local checkout (the build context), not by cloning
# inside the container -- so it contains whatever is in your working tree, subject
# to .dockerignore, rather than whatever is committed. Build from a clean checkout
# if you want the image to match a published commit.
#
# See docs/DOCKER.md for the GPU, Blender and hardware-video caveats.
#
# Base is Ubuntu 24.04 ("noble") because that is what Drake 1.56.0 ships a
# binary tarball for, and its system Python is 3.12 -- which the RB-Y1 IKFast
# extensions require (they build as cpython-312).

FROM ubuntu:24.04

ARG DRAKE_VERSION=1.56.0
ARG BLENDER_VERSION=5.0.1
# sha256 of blender-${BLENDER_VERSION}-linux-x64.tar.xz, as published in
# Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}.sha256. Change it together
# with BLENDER_VERSION, or set it empty to skip the check.
ARG BLENDER_SHA256=8019580ee1b7262e505f4196a00237ccf743c88d205b38d34201510676e60b09
ARG WITH_BLENDER=true
# LaTeX + manim are needed ONLY to rebuild the supplementary video, and pull in
# several GB. Off by default; the experiments and all paper figures do not use them.
ARG WITH_VIDEO=false

ENV DEBIAN_FRONTEND=noninteractive \
    DRAKE_INSTALL_DIR=/opt/drake \
    PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
# cmake/g++/libgtest-dev : the IIWA C++ extension and its test suite
# ffmpeg + fonts-dejavu-core : the video pipeline (drawtext has no font fallback)
# libegl1/libgl1/libxi6/... : headless Blender and Drake's VTK renderer
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget xz-utils git \
        build-essential cmake pkg-config \
        libgtest-dev \
        python3 python3-dev python3-venv python3-pip \
        ffmpeg fonts-dejavu-core \
        libegl1 libgl1 libglu1-mesa libxi6 libxrender1 libxkbcommon0 libsm6 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# LaTeX, only when the video stack is requested: manim renders every equation
# through MathTex, so the explainer scenes cannot build without it.
RUN if [ "$WITH_VIDEO" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        texlive texlive-latex-extra texlive-fonts-recommended dvisvgm \
      && rm -rf /var/lib/apt/lists/* ; \
    fi

# ---------------------------------------------------------------------------
# Drake 1.56.0 -- ONE installation, shared by all three experiments
# ---------------------------------------------------------------------------
# The binary tarball is used rather than the pip wheel because the IIWA
# experiment compiles a pybind11 extension against Drake's C++ library and so
# needs lib/cmake/drake/drake-config.cmake, which the wheel does not carry.
#
# Sharing one Drake across all three environments also makes the pybind11 ABI
# trap structurally impossible: there is no second pydrake anywhere in the
# image for a compiled extension to be imported alongside.
RUN curl -fsSL -o /tmp/drake.tar.gz \
      "https://github.com/RobotLocomotion/drake/releases/download/v${DRAKE_VERSION}/drake-${DRAKE_VERSION}-noble.tar.gz" \
    && tar -xzf /tmp/drake.tar.gz -C /opt \
    && rm /tmp/drake.tar.gz \
    && yes | /opt/drake/share/drake/setup/install_prereqs \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/opt/drake/lib/python3.12/site-packages

# ---------------------------------------------------------------------------
# Blender (optional; needed only for the figures and the video)
# ---------------------------------------------------------------------------
# download.blender.org sits behind a Cloudflare challenge that answers 403 to
# every non-browser client, so it is tried first but cannot be relied on. The
# three fallbacks are Blender's own listed mirrors; the archive is checksummed
# against the officially published sha256 either way, so which one serves it
# does not matter.
RUN if [ "$WITH_BLENDER" = "true" ]; then \
      set -eu; \
      archive="blender-${BLENDER_VERSION}-linux-x64.tar.xz"; \
      series="Blender${BLENDER_VERSION%.*}"; \
      got=""; \
      for base in \
            "https://download.blender.org/release" \
            "https://mirrors.ocf.berkeley.edu/blender/release" \
            "https://mirror.clarkson.edu/blender/release" \
            "https://ftp.nluug.nl/pub/graphics/blender/release" ; do \
        echo "Blender: trying ${base}/${series}/${archive}"; \
        if curl -fsSL --connect-timeout 20 --retry 2 -o /tmp/blender.tar.xz \
             "${base}/${series}/${archive}"; then got="$base"; break; fi; \
      done; \
      if [ -z "$got" ]; then \
        echo "Blender ${BLENDER_VERSION} could not be downloaded from any mirror."; \
        echo "Build with --build-arg WITH_BLENDER=false if you do not need the figures."; \
        exit 1; \
      fi; \
      echo "Blender: downloaded from ${got}"; \
      if [ -n "$BLENDER_SHA256" ]; then \
        echo "${BLENDER_SHA256}  /tmp/blender.tar.xz" | sha256sum -c -; \
      fi; \
      mkdir -p /opt/blender; \
      tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1; \
      rm /tmp/blender.tar.xz; \
      ln -s /opt/blender/blender /usr/local/bin/blender; \
    fi
ENV BLENDER=/usr/local/bin/blender

WORKDIR /release

# ---------------------------------------------------------------------------
# Three virtual environments, one per experiment
# ---------------------------------------------------------------------------
# Separate rather than shared: the UR5e experiment pins numpy and eaik tightly
# and there is no reason to impose those pins on the other two.
#
# `drake` is filtered out of every dependency list -- it is already installed
# at /opt/drake and is on PYTHONPATH. Installing the wheel on top would put a
# second pydrake in the image, which is exactly what must not happen.
#
# Only the dependency manifests are copied here, before the source tree, so that
# editing any source file does not invalidate these installs. They are the slow,
# network-bound layers; keeping them cached is what makes a rebuild quick.
COPY iiwa-bimanual/requirements.txt        /release/iiwa-bimanual/requirements.txt
COPY ur5e-grasp-selection/requirements.txt /release/ur5e-grasp-selection/requirements.txt
COPY rby1-whole-body/pyproject.toml        /release/rby1-whole-body/pyproject.toml

RUN python3 -m venv /opt/venv/iiwa \
    && /opt/venv/iiwa/bin/pip install --no-cache-dir -U pip \
    && /opt/venv/iiwa/bin/pip install --no-cache-dir -r /release/iiwa-bimanual/requirements.txt

RUN python3 -m venv /opt/venv/ur5e \
    && /opt/venv/ur5e/bin/pip install --no-cache-dir -U pip \
    && grep -v '^drake' /release/ur5e-grasp-selection/requirements.txt > /tmp/ur5e-req.txt \
    && /opt/venv/ur5e/bin/pip install --no-cache-dir -r /tmp/ur5e-req.txt

RUN python3 -m venv /opt/venv/rby1 \
    && /opt/venv/rby1/bin/pip install --no-cache-dir -U pip setuptools wheel \
    && python3 -c "import tomllib; d=tomllib.load(open('/release/rby1-whole-body/pyproject.toml','rb'))['project']; \
deps=list(d['dependencies']) + (list(d.get('optional-dependencies',{}).get('video',[])) if '${WITH_VIDEO}'=='true' else []); \
print('\\n'.join(x for x in deps if not x.lower().startswith('drake')))" > /tmp/rby1-req.txt \
    && /opt/venv/rby1/bin/pip install --no-cache-dir -r /tmp/rby1-req.txt

# ---------------------------------------------------------------------------
# Pre-build the IIWA C++ extension
# ---------------------------------------------------------------------------
# Copied on its own, before the rest of the tree: this is an LTO build of several
# minutes, and isolating it means editing a README or a Python script does not
# trigger a recompile.
#
# OPTIMIZED_BUILD=ON is what the paper's runtime numbers were produced with.
# -march=native is deliberately NOT used: it breaks Eigen alignment against
# pybind11, and an image must run on hosts other than the one that built it.
COPY iiwa-bimanual/cpp_parameterization /release/iiwa-bimanual/cpp_parameterization
RUN cmake -S /release/iiwa-bimanual/cpp_parameterization/cpp \
          -B /release/iiwa-bimanual/cpp_parameterization/build \
          -DCMAKE_PREFIX_PATH=/opt/drake -DOPTIMIZED_BUILD=ON \
    && cmake --build /release/iiwa-bimanual/cpp_parameterization/build -j"$(nproc)"

# The RB-Y1 editable install compiles the two IKFast extensions
# (rainbow_{left,right}_arm_ik) -- seconds, not minutes. --no-deps keeps the
# pinned drake wheel out; its runtime deps were installed above from pyproject.
COPY rby1-whole-body/setup.py              /release/rby1-whole-body/setup.py
COPY rby1-whole-body/cpp_parameterization  /release/rby1-whole-body/cpp_parameterization
COPY rby1-whole-body/src                   /release/rby1-whole-body/src
RUN cd /release/rby1-whole-body && /opt/venv/rby1/bin/pip install --no-cache-dir --no-deps -e .

# Everything else. Build artifacts produced above survive this: .dockerignore
# excludes *.so and the build trees, so the host copy cannot overwrite them.
COPY . /release

# Install the vendored Meshcat->Blender add-on so the figure pipelines work.
RUN if [ "$WITH_BLENDER" = "true" ]; then \
      /opt/venv/iiwa/bin/python /release/iiwa-bimanual/scripts/figures/install_meshcat_importer.py || true ; \
    fi

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
