# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PROFILE_ID
ARG PROFILE_STATUS
ENV AMD_AI_PROFILE_ID="${PROFILE_ID}" \
    AMD_AI_PROFILE_STATUS="${PROFILE_STATUS}"

USER root
WORKDIR /workspace
COPY torch-constraints.txt requirements.lock /opt/amd-ai/project-locks/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    if [ -s /opt/amd-ai/project-locks/requirements.lock ]; then \
      uv pip install \
        --python /opt/venv/bin/python \
        --constraint /opt/amd-ai/project-locks/torch-constraints.txt \
        --require-hashes \
        --requirements /opt/amd-ai/project-locks/requirements.lock; \
    fi \
 && /opt/venv/bin/python /opt/amd-ai/torch-manifest.py verify \
      /opt/amd-ai/torch-manifest.json

COPY --chown=1000:1000 . /workspace
COPY project-entrypoint /usr/local/bin/project-entrypoint
RUN chmod 0755 /usr/local/bin/project-entrypoint \
 && rm -rf /root/.cache/uv

USER 1000:1000
ENTRYPOINT ["/usr/local/bin/project-entrypoint"]
CMD ["bash"]
