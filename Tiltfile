# Tiltfile — minimal: just docker build + app deploy
# Infrastructure deployed separately via: task helm:infra

load("ext://helm_resource", "helm_resource")

docker_repo = "registry.localhost:5000"
image_name = "rag-server"
image_tag = "tilt-dev"
full_image = docker_repo + "/" + image_name + ":" + image_tag

docker_build(
    full_image,
    context=".",
    dockerfile="Dockerfile",
    build_args={
        "STAGE": "development",
        "BUILD_TARGET": "cpu",
        "SOURCE_VERSION": "tilt-dev",
    },
    target="development-cpu",
    live_update=[
        sync("./server/app", "/code/project/app"),
        sync("./server/entrypoint.sh", "/code/project/entrypoint.sh"),
    ],
)

k8s_yaml(helm(
    "./deploy/helm/rag-app",
    name="rag-app",
    values=["./deploy/helm/rag-app/values-dev.yaml"],
    set=[
        "image.repository=" + docker_repo + "/" + image_name,
        "image.tag=" + image_tag,
        "image.pullPolicy=IfNotPresent",
        "config.logLevel=DEBUG",
        "server.replicas=1",
        "server.pdb.enabled=false",
    ],
))

k8s_resource(
    "rag-app-server",
    port_forwards="8001:8001",
)
